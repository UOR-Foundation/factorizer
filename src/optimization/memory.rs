//! Memory optimization utilities
//!
//! This module provides memory pooling and efficient allocation strategies
//! to reduce memory fragmentation and improve cache locality.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// Memory pool for reusable allocations
pub struct MemoryPool<T> {
    pool: Arc<Mutex<VecDeque<Box<T>>>>,
    max_size: usize,
    initializer: Arc<dyn Fn() -> T + Send + Sync>,
}

impl<T: Default + Send + 'static> MemoryPool<T> {
    /// Create new memory pool with default initializer
    pub fn new(max_size: usize) -> Self {
        MemoryPool {
            pool: Arc::new(Mutex::new(VecDeque::with_capacity(max_size))),
            max_size,
            initializer: Arc::new(|| T::default()),
        }
    }
}

impl<T> std::fmt::Debug for MemoryPool<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryPool")
            .field("max_size", &self.max_size)
            .field("current_size", &self.pool.lock().map(|p| p.len()).unwrap_or(0))
            .finish()
    }
}

impl<T: Send + 'static> MemoryPool<T> {
    /// Create new memory pool with custom initializer
    pub fn with_initializer<F>(max_size: usize, initializer: F) -> Self
    where
        F: Fn() -> T + Send + Sync + 'static,
    {
        MemoryPool {
            pool: Arc::new(Mutex::new(VecDeque::with_capacity(max_size))),
            max_size,
            initializer: Arc::new(initializer),
        }
    }

    /// Acquire object from pool
    pub fn acquire(&self) -> PooledObject<T> {
        let obj = {
            let mut pool = self.pool.lock().unwrap();
            pool.pop_front()
        };

        let value = match obj {
            Some(boxed) => boxed,
            None => Box::new((self.initializer)()),
        };

        PooledObject {
            value: Some(value),
            pool: Arc::clone(&self.pool),
            max_size: self.max_size,
        }
    }

    /// Clear the pool
    pub fn clear(&self) {
        let mut pool = self.pool.lock().unwrap();
        pool.clear();
    }

    /// Get current pool size
    pub fn size(&self) -> usize {
        self.pool.lock().unwrap().len()
    }
}

/// RAII wrapper for pooled objects
#[derive(Debug)]
pub struct PooledObject<T> {
    value: Option<Box<T>>,
    pool: Arc<Mutex<VecDeque<Box<T>>>>,
    max_size: usize,
}

impl<T> PooledObject<T> {
    /// Get reference to the value
    pub fn get(&self) -> &T {
        self.value.as_ref().unwrap()
    }

    /// Get mutable reference to the value
    pub fn get_mut(&mut self) -> &mut T {
        self.value.as_mut().unwrap()
    }
}

impl<T> Drop for PooledObject<T> {
    fn drop(&mut self) {
        if let Some(value) = self.value.take() {
            let mut pool = self.pool.lock().unwrap();
            if pool.len() < self.max_size {
                pool.push_back(value);
            }
        }
    }
}

impl<T> std::ops::Deref for PooledObject<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

impl<T> std::ops::DerefMut for PooledObject<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.get_mut()
    }
}

/// Arena allocator for batch allocations
#[derive(Debug)]
pub struct Arena {
    chunks: RefCell<Vec<Vec<u8>>>,
    current: RefCell<Vec<u8>>,
    chunk_size: usize,
}

impl Arena {
    /// Create new arena with specified chunk size
    pub fn new(chunk_size: usize) -> Self {
        Arena {
            chunks: RefCell::new(Vec::new()),
            current: RefCell::new(Vec::with_capacity(chunk_size)),
            chunk_size,
        }
    }

    /// Allocate bytes from arena
    pub fn alloc(&self, size: usize) -> *mut u8 {
        let mut current = self.current.borrow_mut();

        if current.capacity() - current.len() < size {
            // Need new chunk
            let mut chunks = self.chunks.borrow_mut();
            let old_chunk =
                std::mem::replace(&mut *current, Vec::with_capacity(self.chunk_size.max(size)));
            if !old_chunk.is_empty() {
                chunks.push(old_chunk);
            }
        }

        let ptr = unsafe { current.as_mut_ptr().add(current.len()) };

        current.extend(std::iter::repeat(0).take(size));
        ptr
    }

    /// Clear arena, keeping allocated memory for reuse
    pub fn clear(&self) {
        self.current.borrow_mut().clear();
        self.chunks.borrow_mut().clear();
    }
}

/// Compressed bit vector for memory-efficient boolean arrays
#[derive(Debug)]
pub struct BitVector {
    bits: Vec<u64>,
    len: usize,
}

impl BitVector {
    /// Create new bit vector with given length
    pub fn new(len: usize) -> Self {
        let num_words = (len + 63) / 64;
        BitVector {
            bits: vec![0; num_words],
            len,
        }
    }

    /// Set bit at index
    #[inline]
    pub fn set(&mut self, index: usize, value: bool) {
        debug_assert!(index < self.len);
        let word = index / 64;
        let bit = index % 64;

        if value {
            self.bits[word] |= 1u64 << bit;
        } else {
            self.bits[word] &= !(1u64 << bit);
        }
    }

    /// Get bit at index
    #[inline]
    pub fn get(&self, index: usize) -> bool {
        debug_assert!(index < self.len);
        let word = index / 64;
        let bit = index % 64;
        (self.bits[word] >> bit) & 1 != 0
    }

    /// Count set bits
    pub fn count_ones(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Clear all bits
    pub fn clear(&mut self) {
        self.bits.fill(0);
    }
}

/// Memory-mapped large arrays for out-of-core computation
#[derive(Debug)]
pub struct MappedArray<T> {
    data: memmap2::MmapMut,
    len: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Copy> MappedArray<T> {
    /// Create new memory-mapped array
    pub fn new(len: usize) -> std::io::Result<Self> {
        use memmap2::MmapOptions;

        let size = len * std::mem::size_of::<T>();
        let file = tempfile::tempfile()?;
        file.set_len(size as u64)?;

        let mmap = unsafe { MmapOptions::new().map_mut(&file)? };

        Ok(MappedArray {
            data: mmap,
            len,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Get element at index
    pub fn get(&self, index: usize) -> Option<T> {
        if index >= self.len {
            return None;
        }

        unsafe {
            let ptr = self.data.as_ptr() as *const T;
            Some(*ptr.add(index))
        }
    }

    /// Set element at index
    pub fn set(&mut self, index: usize, value: T) {
        debug_assert!(index < self.len);

        unsafe {
            let ptr = self.data.as_mut_ptr() as *mut T;
            *ptr.add(index) = value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_pool() {
        let pool: MemoryPool<Vec<u8>> = MemoryPool::new(10);

        let mut obj1 = pool.acquire();
        obj1.extend_from_slice(b"hello");
        assert_eq!(&**obj1, b"hello");

        drop(obj1);
        assert_eq!(pool.size(), 1);

        let obj2 = pool.acquire();
        assert_eq!(pool.size(), 0); // Reused from pool
    }

    #[test]
    fn test_bit_vector() {
        let mut bits = BitVector::new(100);

        bits.set(10, true);
        bits.set(50, true);
        bits.set(99, true);

        assert!(bits.get(10));
        assert!(!bits.get(11));
        assert!(bits.get(50));
        assert!(bits.get(99));

        assert_eq!(bits.count_ones(), 3);
    }
}

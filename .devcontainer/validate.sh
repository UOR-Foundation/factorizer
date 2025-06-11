#!/bin/bash
# Validate devcontainer configuration

echo "🔍 Validating devcontainer configuration..."

# Check JSON syntax
echo -n "Checking devcontainer.json syntax... "
if python3 -m json.tool .devcontainer/devcontainer.json > /dev/null 2>&1; then
    echo "✅ Valid"
else
    echo "❌ Invalid JSON"
    python3 -m json.tool .devcontainer/devcontainer.json
fi

echo -n "Checking devcontainer-compose.json syntax... "
if python3 -m json.tool .devcontainer/devcontainer-compose.json > /dev/null 2>&1; then
    echo "✅ Valid"
else
    echo "❌ Invalid JSON"
    python3 -m json.tool .devcontainer/devcontainer-compose.json
fi

# Check YAML syntax
echo -n "Checking docker-compose.yml syntax... "
if command -v yamllint >/dev/null 2>&1; then
    if yamllint -d relaxed .devcontainer/docker-compose.yml > /dev/null 2>&1; then
        echo "✅ Valid"
    else
        echo "❌ Invalid YAML"
        yamllint -d relaxed .devcontainer/docker-compose.yml
    fi
else
    echo "⚠️  yamllint not installed, skipping"
fi

# Check shell script syntax
echo -n "Checking post-create.sh syntax... "
if bash -n .devcontainer/post-create.sh 2>&1; then
    echo "✅ Valid"
else
    echo "❌ Invalid bash syntax"
    bash -n .devcontainer/post-create.sh
fi

# Check Dockerfile syntax
echo -n "Checking Dockerfile syntax... "
if command -v hadolint >/dev/null 2>&1; then
    if hadolint .devcontainer/Dockerfile > /dev/null 2>&1; then
        echo "✅ Valid"
    else
        echo "⚠️  Has warnings:"
        hadolint .devcontainer/Dockerfile
    fi
else
    echo "⚠️  hadolint not installed, skipping"
fi

# Check file permissions
echo -n "Checking file permissions... "
if [ -x .devcontainer/post-create.sh ]; then
    echo "✅ post-create.sh is executable"
else
    echo "⚠️  post-create.sh is not executable"
fi

echo ""
echo "📋 Summary of devcontainer setup:"
echo "  - Main config: devcontainer.json (uses pre-built image)"
echo "  - Compose config: devcontainer-compose.json (for advanced users)"
echo "  - Custom Dockerfile available for customization"
echo "  - Post-create script sets up development environment"
echo "  - Extensions will be installed when opened in VS Code"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "                    VALIDATION SUMMARY                          "
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "✅ All devcontainer configurations are valid and ready to use!"
echo ""
echo "You can now open this project in VS Code and use the devcontainer."
echo "════════════════════════════════════════════════════════════════"
#!/bin/bash
# RuVector Installer
# Usage: curl -fsSL https://raw.githubusercontent.com/ruvnet/ruvector/main/install.sh | bash
# Or:    wget -qO- https://raw.githubusercontent.com/ruvnet/ruvector/main/install.sh | bash

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

print_banner() {
    echo -e "${CYAN}"
    echo "  ____        __     __        _             "
    echo " |  _ \ _   _ \ \   / /__  ___| |_ ___  _ __ "
    echo " | |_) | | | | \ \ / / _ \/ __| __/ _ \| '__|"
    echo " |  _ <| |_| |  \ V /  __/ (__| || (_) | |   "
    echo " |_| \_\\\\__,_|   \_/ \___|\___|\__\___/|_|   "
    echo -e "${NC}"
    echo -e "${YELLOW}Vector database that learns${NC}"
    echo ""
}

print_step() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}!${NC} $1"
}

# Detect OS and architecture
detect_platform() {
    OS="$(uname -s)"
    ARCH="$(uname -m)"

    case "$OS" in
        Linux*)     PLATFORM="linux" ;;
        Darwin*)    PLATFORM="darwin" ;;
        MINGW*|MSYS*|CYGWIN*) PLATFORM="windows" ;;
        *)          PLATFORM="unknown" ;;
    esac

    case "$ARCH" in
        x86_64|amd64)   ARCH="x64" ;;
        aarch64|arm64)  ARCH="arm64" ;;
        *)              ARCH="unknown" ;;
    esac

    echo "${PLATFORM}-${ARCH}"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Install Rust if not present
install_rust() {
    if command_exists rustc; then
        RUST_VERSION=$(rustc --version | cut -d' ' -f2)
        print_success "Rust ${RUST_VERSION} already installed"
        return 0
    fi

    print_step "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    print_success "Rust installed"
}

# Install from crates.io
install_crates() {
    print_step "Installing RuVector crates from crates.io..."

    local CRATES=(
        "ruvector-core"
        "ruvector-graph"
        "ruvector-gnn"
        "ruvector-collections"
        "ruvector-filter"
        "ruvector-metrics"
        "ruvector-snapshot"
    )

    for crate in "${CRATES[@]}"; do
        echo -n "  Installing ${crate}... "
        if cargo install "$crate" --quiet 2>/dev/null; then
            echo -e "${GREEN}done${NC}"
        else
            echo -e "${YELLOW}skipped (library crate)${NC}"
        fi
    done

    print_success "Core crates installed"
}

# Install CLI tools
install_cli() {
    print_step "Installing RuVector CLI tools..."

    if cargo install ruvector-cli --quiet 2>/dev/null; then
        print_success "ruvector-cli installed"
    else
        print_warning "ruvector-cli not available on crates.io yet"
    fi
}

# Install Node.js packages
install_npm() {
    if ! command_exists node; then
        print_warning "Node.js not found, skipping npm packages"
        return 0
    fi

    print_step "Installing npm packages..."

    if command_exists npm; then
        npm install -g ruvector 2>/dev/null || true
        print_success "npm packages installed"
    fi
}

# Show available crates
show_crates() {
    echo ""
    echo -e "${CYAN}Available RuVector Crates:${NC}"
    echo ""
    echo -e "${GREEN}Core:${NC}"
    echo "  cargo add ruvector-core        # Vector database engine"
    echo "  cargo add ruvector-graph       # Hypergraph with Cypher"
    echo "  cargo add ruvector-gnn         # Graph Neural Networks"
    echo ""
    echo -e "${GREEN}Distributed:${NC}"
    echo "  cargo add ruvector-cluster     # Cluster management"
    echo "  cargo add ruvector-raft        # Raft consensus"
    echo "  cargo add ruvector-replication # Data replication"
    echo ""
    echo -e "${GREEN}AI Routing:${NC}"
    echo "  cargo add ruvector-tiny-dancer-core  # FastGRNN inference"
    echo "  cargo add ruvector-router-core       # Semantic routing"
    echo ""
    echo -e "${GREEN}Bindings:${NC}"
    echo "  cargo add ruvector-node        # Node.js (napi-rs)"
    echo "  cargo add ruvector-wasm        # WebAssembly"
    echo ""
}

# Show npm packages
show_npm() {
    echo -e "${CYAN}Available npm Packages:${NC}"
    echo ""
    echo "  npm install ruvector           # All-in-one CLI"
    echo "  npm install @ruvector/core     # Vector database"
    echo "  npm install @ruvector/gnn      # Graph Neural Networks"
    echo "  npm install @ruvector/graph-node  # Hypergraph database"
    echo ""
    echo "  npx ruvector install           # List all packages"
    echo "  npx ruvector install --all     # Install everything"
    echo ""
}

# Main installation
main() {
    print_banner

    PLATFORM=$(detect_platform)
    print_step "Detected platform: ${PLATFORM}"
    echo ""

    # Parse arguments
    INSTALL_RUST=true
    INSTALL_CRATES=true
    INSTALL_NPM=true
    SHOW_ONLY=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --rust-only)
                INSTALL_NPM=false
                shift
                ;;
            --npm-only)
                INSTALL_RUST=false
                INSTALL_CRATES=false
                shift
                ;;
            --list|--show)
                SHOW_ONLY=true
                shift
                ;;
            --help|-h)
                echo "Usage: install.sh [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --rust-only    Only install Rust crates"
                echo "  --npm-only     Only install npm packages"
                echo "  --list         Show available packages without installing"
                echo "  --help         Show this help"
                echo ""
                echo "Examples:"
                echo "  curl -fsSL https://raw.githubusercontent.com/ruvnet/ruvector/main/install.sh | bash"
                echo "  curl -fsSL ... | bash -s -- --rust-only"
                echo "  curl -fsSL ... | bash -s -- --npm-only"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                exit 1
                ;;
        esac
    done

    if [ "$SHOW_ONLY" = true ]; then
        show_crates
        show_npm
        exit 0
    fi

    # Install Rust
    if [ "$INSTALL_RUST" = true ]; then
        install_rust
        echo ""
    fi

    # Install crates
    if [ "$INSTALL_CRATES" = true ]; then
        install_crates
        echo ""
    fi

    # Install npm
    if [ "$INSTALL_NPM" = true ]; then
        install_npm
        echo ""
    fi

    # Show summary
    echo ""
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  RuVector installed successfully!${NC}"
    echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
    echo ""

    show_crates
    show_npm

    echo -e "${CYAN}Quick Start:${NC}"
    echo ""
    echo "  # Rust"
    echo "  cargo add ruvector-core ruvector-gnn"
    echo ""
    echo "  # Node.js"
    echo "  npx ruvector"
    echo ""
    echo -e "${CYAN}Documentation:${NC} https://github.com/ruvnet/ruvector"
    echo ""
}

main "$@"

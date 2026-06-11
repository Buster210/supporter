AUTO_APPROVED_BINARIES = {
    "ls",
    "tree",
    "pwd",
    "which",
    "file",
    "wc",
    "grep",
    "rg",
    "head",
    "tail",
    "cat",
    "date",
    "echo",
    "whoami",
    "ps",
    "df",
    "uptime",
    "mkdir",
    "touch",
    "uname",
    "sort",
    "uniq",
    "diff",
    "du",
    "stat",
    "awk",
    "gzip",
}

BLOCKED_BINARIES = {
    "sudo",
    "su",
    "mkfs",
    "fdisk",
    "dd",
    "reboot",
    "shutdown",
    "mount",
    "umount",
    "chown",
    "chattr",
    "shred",
    "eval",
    "exec",
    "source",
    "env",
    "sudoedit",
    "osascript",
    "lldb",
    "dtrace",
    "dtruss",
    "launchctl",
    "defaults",
    "plutil",
    "automator",
    "expect",
    "screen",
    "tmux",
    "script",
    "ssh",
    "scp",
    "sftp",
    "telnet",
    "rsync",
}

NETWORK_EGRESS_BINARIES = {
    "curl",
    "wget",
    "nc",
    "ncat",
    "socat",
    "httpie",
    "http",
    "ftp",
    "tftp",
}


MUTATING_BINARIES = {
    "cp",
    "mv",
    "rm",
    "mkdir",
    "rmdir",
    "touch",
    "tee",
    "dd",
    "chmod",
    "chown",
    "ln",
    "install",
    "patch",
}

WRITE_REDIRECT_TOKENS = {">", ">>"}

SENSITIVE_SYSTEM_PATHS = {
    "/etc",
    "/var/log",
    "~/.ssh",
    "~/.aws",
    "~/.config",
    "~/.gnupg",
    "~/.kube",
    "~/.docker/config.json",
    "~/.bashrc",
    "~/.zshrc",
    "~/.profile",
}

TRUSTED_EXECUTABLE_PATH_PREFIXES = [
    "/usr/bin",
    "/bin",
    "/usr/local/bin",
    "/opt/homebrew/bin",
]

SENSITIVE_FILE_PATTERNS = [
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
    "*_rsa",
    "*.p12",
    "*.pfx",
    "*secret*",
    "*token*",
    "*credential*",
    "*.kdbx",
    ".netrc",
    ".npmrc",
    ".pypirc",
    ".aws/credentials",
    ".docker/config.json",
    ".kube/config",
]

SECRET_LITERAL_PATTERNS = [
    r"AIza[0-9A-Za-z\-_]{35}",
    r"sk-[a-zA-Z0-9]{32,}",
    r"AKIA[0-9A-Z]{16}",
    r"ghp_[a-zA-Z0-9]{36}",
]

SECRET_KEYWORD_TRIGGERS = ("password", "secret", "token", "api_key")
SECRET_KEYWORD_PATTERN = (
    r"(?i)(" + "|".join(SECRET_KEYWORD_TRIGGERS) + r")\s*[=:]\s*\S+"
)

NETWORK_UPLOAD_FLAGS = {
    "-F",
    "--form",
    "-T",
    "--upload-file",
    "--post-file",
    "--body-file",
}

PACKAGE_MANAGER_BINARIES = {
    "npm",
    "yarn",
    "pnpm",
    "bun",
    "pip",
    "uv",
    "poetry",
    "cargo",
    "go",
    "gem",
}

AUTO_APPROVED_GIT_SUBCOMMANDS = {
    "branch",
    "describe",
    "diff",
    "log",
    "ls-files",
    "remote",
    "rev-parse",
    "shortlog",
    "show",
    "status",
}

INLINE_PAYLOAD_INTERPRETER_BINARIES = {
    "python",
    "python3",
    "node",
    "js",
    "bash",
    "sh",
    "perl",
    "ruby",
}

RM_BLOCKED_TARGET_PATHS = {"/", "/usr", "/bin", "/etc", "/var", "/home", "/root"}

SHELL_INTERPRETER_BINARIES = {
    "sh",
    "bash",
    "zsh",
    "dash",
    "fish",
    "python",
    "python3",
    "node",
    "perl",
    "ruby",
}

FILE_READING_BINARIES = {
    "cat",
    "tail",
    "head",
    "grep",
    "rg",
    "file",
    "wc",
    "tar",
    "zip",
}

UNTRUSTED_EXECUTION_TEMP_DIRS = [
    "/tmp",  # noqa: S108
    "/private/tmp",
    "/var/folders",
    "/private/var/folders",
]

PACKAGE_INSTALL_SUBCOMMANDS = {"install", "i", "ci", "add", "sync"}

RISKY_PYTHON_SYMBOL_NAMES = {
    "getattr",
    "setattr",
    "__import__",
    "compile",
    "globals",
    "locals",
    "vars",
    "chr",
    "ord",
    "exec",
    "eval",
    "import_module",
    "__builtins__",
}

RISKY_PYTHON_ATTRIBUTE_NAMES = {
    "__import__",
    "__builtins__",
    "__globals__",
    "__dict__",
    "__class__",
    "__subclasses__",
    "__bases__",
    "__mro__",
    "import_module",
}

CONFIRMATION_REQUIRED_PYTHON_MODULES = {
    "os",
    "subprocess",
    "socket",
    "importlib",
    "urllib",
    "http",
    "httplib",
    "httpx",
    "requests",
    "aiohttp",
    "websocket",
    "websockets",
}

BLOCKED_PYTHON_MODULES = {
    "codecs",
    "base64",
    "binascii",
    "marshal",
    "pickle",
    "zlib",
    "bz2",
    "lzma",
}

CONFIRMATION_REQUIRED_BINARIES = {
    "chmod",
    "mv",
    "rm",
    "rmdir",
    "kill",
    "pkill",
    "nice",
    "renice",
    "top",
    "htop",
    "lsof",
    "ssh-keygen",
    "curl",
    "wget",
    "xargs",
    "parallel",
}

# Security Tiers
TIER_SAFE = 1
TIER_CONFIRM = 2
TIER_BLOCK = 3

# Resource Limits
CPU_LIMIT_SEC = 10
MEM_LIMIT_BYTES = 1024 * 1024 * 1024

# Timeouts
EXECUTION_TIMEOUT_SEC = 30

# Buffers
OUTPUT_BUFFER_LIMIT = 100 * 1024

# Network Data Prefixes
NETWORK_DATA_PREFIXES = [
    "-d@",
    "--data@",
    "-d @",
    "--data @",
    "--data-binary @",
    "--data-urlencode @",
]

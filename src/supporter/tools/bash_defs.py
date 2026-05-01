TIER1_BINARIES = {
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
    "cp",
    "uname",
    "env",
    "sort",
    "uniq",
    "diff",
    "find",
    "du",
    "stat",
    "sed",
    "awk",
    "tar",
    "zip",
    "unzip",
    "gzip",
    "ln",
}

TIER3_BINARIES = {
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

NETWORK_BINARIES = {
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

SHELL_METACHARACTERS = {
    "|",
    ">",
    ">>",
    "<",
    "&",
    ";",
    "$",
    "~",
    "`",
    "$(",
    ")",
    "(",
    "&&",
    "||",
}

SYSTEM_DIRECTORIES = {
    "/etc",
    "/var/log",
    "~/.ssh",
    "~/.bashrc",
    "~/.zshrc",
    "~/.profile",
}

TRUSTED_PREFIXES = ["/usr/bin", "/bin", "/usr/local/bin", "/opt/homebrew/bin"]

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

SECRET_PATTERNS = [
    r"AIza[0-9A-Za-z\-_]{35}",
    r"sk-[a-zA-Z0-9]{32,}",
    r"AKIA[0-9A-Z]{16}",
    r"ghp_[a-zA-Z0-9]{36}",
    r"(?i)(password|secret|token|api_key)\s*[=:]\s*\S+",
]


UPLOAD_FLAGS = {
    "-F",
    "--form",
    "-T",
    "--upload-file",
    "--post-file",
    "--body-file",
}

PACKAGE_MANAGERS = {
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

TIER1_GIT_SUBCOMMANDS = {
    "add",
    "branch",
    "describe",
    "diff",
    "fetch",
    "log",
    "ls-files",
    "remote",
    "rev-parse",
    "shortlog",
    "show",
    "stash",
    "status",
    "commit",
    "checkout",
    "merge",
    "push",
    "pull",
}

INTERPRETERS = {"python", "python3", "node", "js", "bash", "sh", "perl", "ruby"}

RM_NUCLEAR_PATHS = {"/", "/usr", "/bin", "/etc", "/var", "/home", "/root"}

SHELL_BINS = {
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

FILE_READING_BINS = {
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

TEMP_DIRS = ["/tmp", "/private/tmp", "/var/folders", "/private/var/folders"]  # noqa: S108

INSTALL_CMDS = {"install", "i", "ci", "add", "sync"}

RISKY_PYTHON_NAMES = {
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

RISKY_PYTHON_ATTRS = {
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

RISKY_PYTHON_MODULES = {
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

TIER3_PYTHON_MODULES = {
    "codecs",
    "base64",
    "binascii",
    "marshal",
    "pickle",
    "zlib",
    "bz2",
    "lzma",
}

HIGH_RISK_TIER2_BINARIES = {
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

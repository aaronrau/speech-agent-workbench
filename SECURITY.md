# Security

This project controls local input devices and can paste/type into active
applications. Review changes carefully before running them with access to
`/dev/input`, `/dev/uinput`, or clipboard tools.

## Reporting Issues

Please report security-sensitive issues privately to the repository owner rather
than opening a public issue.

## Local Data

Do not commit:

- `config.json`
- downloaded model files
- generated audio samples
- local cache directories
- credentials or private endpoint URLs

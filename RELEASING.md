# Release process

This repository contains the package:

- **sendnn-inference** (tags: `vX.Y.Z`) — the main Spyre hardware plugin

The package follows [Semantic Versioning](https://semver.org/): use **patch** for bug fixes, **minor** for new features, **major** for breaking changes.

## Release steps

1. Go to the [Releases page](https://github.com/torch-spyre/sendnn-inference/releases) and click "Draft a new release"
2. Click "Choose a tag" and type the new version:
   - For sendnn-inference: `vX.Y.Z` (e.g., `v1.2.0`)
3. Click "Generate release notes" and select the previous tag **of the same package** as the base:
   - For sendnn-inference: select a `v*` tag
4. Review the changes to decide on the version bump, edit release notes, then publish
5. The appropriate workflow will automatically trigger and publish to PyPI

**Alternative**: Create the tag locally first, then create the release on GitHub.

## Test releases

**sendnn-inference**: Automatically published to [test.pypi.org](https://test.pypi.org/project/sendnn-inference/) on every push to main.

**Note**: setuptools_scm automatically converts SemVer pre-release tags to [PEP 440](https://peps.python.org/pep-0440/) format (e.g., `v1.0.0-rc.1` → `1.0.0rc1`).

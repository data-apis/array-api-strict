# Releasing

To release array-api-strict:

- [ ] Create a release branch and make a PR on GitHub.

- [ ] Update `changelog.md` with the changes for the release.

- [ ] Make sure the CI is passing on the release branch PR. Also double check that
  you have properly pulled `main` and merged it into the release branch so
  that the branch contains all the necessary changes for the release.

- [ ] When you are ready to make the release, create a tag with the release number

  ```
  git tag -a 2.2 -m "array-api-strict 2.2"
  ```

  and push it up to GitHub

  ```
  git push origin --tags
  ```

  This will trigger the `publish-package` build on GitHub Actions. Make sure
  that build works correctly and pushes the release up to PyPI. If something
  goes wrong, you may need to delete the tag from GitHub and try again.

  Note that the `array_api_strict.__version__` version as well as the version
  in the package metadata is all automatically computed from the tag, so it is
  not necessary to update the version anywhere else.

- [ ] Once the release is published, you can merge the PR.

- [ ] The conda-forge bot will automatically send a PR to the
  [array-api-strict-feedstock](https://github.com/conda-forge/array-api-strict-feedstock)
  updating the version, which you should merge.

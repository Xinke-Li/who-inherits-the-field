# Publishing steps for the author

The repository and the archive are prepared. What follows records what is already
done and the few click-through steps that remain, because they create public
records under your name.

## Already done

The paper, `README.md`, `CITATION.cff`, and the datasheet carry the public
repository URL (https://github.com/Xinke-Li/who-inherits-the-field) and the
reserved DOI (10.5281/zenodo.21501632). The public branch `main` is a
single-commit history authored and committed by Xinke Li alone, so the GitHub
contributor list shows one person. The full internal history stays on the local
`master` branch, which is never pushed. The key scan
(`grep -rn "$OPENALEX_API_KEY" --exclude-dir=.git .`) and the attribution scan
(no automated-tool attribution anywhere in the tree or the commit) ran clean
before the push; `.gitignore` excludes `.openalex_key`, `*.key`, `.env`, and
`*.zip`.

## Zenodo

Upload `zenodo_archive.zip` (repository root; data, datasheet, and the
reproduction script; the file list is inside the archive) to the draft deposit
that reserved DOI 10.5281/zenodo.21501632, then press publish when the paper is
submitted. Set the metadata by hand: creator is Xinke Li (University of Chicago,
ORCID 0009-0001-0403-3606), the license follows the data sources (OpenAlex
records are CC0, the derived tables are CC BY 4.0 following the genealogy
terms), and the version is 1.0.0.

## GitHub web settings

Two settings live only in the web interface. In the repository's About box, a
one-line description such as: "Five-discipline leakage-audited benchmark for
intellectual inheritance in doctoral training (KDD 2027 D&B submission)".
Suggested topics: `benchmark`, `dataset`, `science-of-science`, `data-leakage`,
`graph-neural-networks`, `kdd`.

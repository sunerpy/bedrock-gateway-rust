# docs/site

The pages of the bedrock-gateway website, published at <https://firlab.app/bedrock-gateway/>. This file is for people
who edit them; it is not itself a page of the site.

The words live here, next to the code they describe, so a change that alters what a user sees updates the page in the
same pull request. The site itself, its theme, components, build and deployment, lives in
[sunerpy/firlab](https://github.com/sunerpy/firlab) under `bedrock-gateway/`. Its sync script copies these pages and
the references they link into the site; do not edit the copies there.

## What is published

| Source                                                                         | Published at                                                                                    |
| ------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------- |
| `docs/site/<path>.md`                                                          | `/bedrock-gateway/<path>`                                                                       |
| `docs/site/zh/<path>.md`                                                       | `/bedrock-gateway/zh/<path>`                                                                    |
| `docs/deploy/docker.md`, `ecs.md`, `lambda.md`, `migration-from-aws-samples.md` | `/bedrock-gateway/operate/docker`, `/operate/ecs`, `/operate/lambda`, `/operate/migration`, English only |
| `docs/openai-protocol-compatibility.md`                                        | `/bedrock-gateway/reference/protocol`, English only                                             |
| `docs/caching-and-reasoning.md`                                                | `/bedrock-gateway/zh/reference/caching-and-reasoning`, Chinese only                             |
| `docs/site/public/bedrock-gateway-logo.svg`                                    | The site's logo and favicon                                                                     |

A reference in one language only gets a short page in the other language that links to it. The sync script in FirLab
(`bedrock-gateway/scripts/sync-bedrock-gateway-docs.sh`) holds the exact mapping. Adding a reference to the site means
adding it there and to the sidebars in the same FirLab change.

## Rules the sync checks

The sync stops before it writes anything when:

- a page exists in one language only: every `docs/site/<path>.md` needs `docs/site/zh/<path>.md`, and the reverse;
- a page sits where a synced reference goes, such as `docs/site/operate/docker.md`;
- a page uses a component the theme does not register (see below);
- a guide page uses a colloquial word (`just`, `simply`, `stuff`; `搞定`, `折腾`, `试试`, `看看` and the like) or names
  the source tree (`src/server/…`); `developers.md` may name the source tree.

## Writing

- Write for someone who runs bedrock-gateway, not for someone who works on it. Say what to do and what happens; leave
  the internals to the references and `developers.md`.
- Commands, field names, routes, error codes, environment variables and model IDs go in code spans.
- Every number, version and output on a page comes from the code, a release or a recorded run. Output shown as a
  program's output is copied from that program, not typed by hand.
- Leave out how a page was checked: no "validated with", "recorded on" or test notes. That evidence belongs in the pull
  request.
- Do not name private endpoints, account IDs or internal resource names. Examples use `localhost:8080`, the AWS
  documentation account `123456789012` and placeholders in angle brackets.
- Chinese pages are written in Chinese, not translated word for word, and keep the English names of commands, fields
  and products.

## Links

Link other files by their real path in this repository, the way GitHub resolves it. A page at the top of `docs/site/`
writes `[Protocol compatibility](../openai-protocol-compatibility.md)` and `[Models](guide/models.md)`; a page in a
subdirectory adds one `../` per level, and a page under `zh/` one more. The sync turns a link to a published reference
into a link to its page on the site, and any other link that leaves the published files into a GitHub link.

Do not write site paths such as `/guide/install` in Markdown links: GitHub cannot follow them. The home page's
frontmatter is the exception, because it is data for the theme, not Markdown. An anchor is the GitHub slug of the
heading it points to; on a Chinese page that is the Chinese heading.

## Components

Pages are GitHub Markdown, plus these components of the site's theme: `HomeIndex`, `HomeSteps`, `SplitBlock`,
`HomePlatforms`, `HomePrivacy`, `HomeScope`, `StatusTag` and VitePress's `Badge`. They render on the site only; on
GitHub a page shows them as plain tags, so keep them to the two home pages.

## The home page

`index.md` and `zh/index.md` carry their content in the `home:` frontmatter, which the theme lays out. The build fails
when a field is missing, misspelt or of the wrong shape (`theme/data/home-schema.ts` in FirLab).

| Field                       | Holds                                                                                                                                                                                 |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `facts`                     | Two to four `term` and `text` rows under the tagline.                                                                                                                                 |
| `visual`                    | The terminal session in the hero: a `label` for screen readers, the `transcript` lines (`command`, `continuation` for a command's further lines, `output`) and an optional `caption`. |
| `index`                     | Groups of features, each with `title`, `body`, `status` (`available` or `opt-in`) and an optional `link`.                                                                             |
| `steps`                     | Four steps, each with `title`, `body` and an optional `command`.                                                                                                                      |
| `routes`, `clients`, `models` | The tables beside the three splits: `columns`, `rows` of `cells`, `code` (the indexes of the columns shown as code) and an optional `caption`.                                     |
| `platforms`                 | The platform table: `columns`, and `rows` with `name`, `status` and one cell per remaining column.                                                                                    |
| `privacy`                   | What goes where: `sendsLabel` and `modes` with `name`, `sends` and `detail`.                                                                                                          |
| `scope`                     | What bedrock-gateway does not do, as `items`.                                                                                                                                         |

In a YAML flow list such as `cells: [a, b, c]` a comma separates cells, so quote a cell that contains one:
`"Cohere, Titan and Nova"`. A code cell after the first column wraps between its words, so a list of model IDs is
written with spaces between them and no commas.

The transcript is a real run of a released version against the default address, `http://localhost:8080/api/v1`, so it
needs no caption. To record a new one, start a gateway with a key of its own, run the commands exactly as the page
shows them and copy the output.

## Preview

With a FirLab checkout next to this repository:

```sh
cd ../firlab/bedrock-gateway
pnpm install --frozen-lockfile
./scripts/sync-bedrock-gateway-docs.sh ../../bedrock-gateway-rust
pnpm dev    # http://localhost:5173/bedrock-gateway/
```

`pnpm build` followed by `./scripts/check-dist.sh dist` runs the same checks as the deployment.

## Publishing

Two workflows connect this repository to the site:

- `.github/workflows/docs-site.yml` runs on pull requests that touch the pages or the published references. It syncs
  them into a checkout of FirLab's `main`, builds the site and runs `check-dist.sh`, without a secret. It is advisory
  and not part of `CI Success`.
- `.github/workflows/publish-site.yml` runs after a merge to `main` that touches them. It runs the same sync, commits
  the result as `docs(bedrock-gateway): sync from bedrock-gateway-rust@<sha>` to the branch `bedrock-gateway-docs/sync`
  in FirLab, and opens a pull request from it, or updates the one already open. Once FirLab's checks pass, it
  squash-merges the pull request at the commit that was checked, and FirLab's deploy publishes the site. A failed
  check leaves the pull request open; the next run replaces its commit.

`publish-site.yml` needs the repository secret `FIRLAB_DOCS_TOKEN`: a fine-grained personal access token for
`sunerpy/firlab` only, with Contents and Pull requests read and write and nothing else. GitHub has no API that creates
one, so it is made by hand and stored with:

```sh
gh secret set FIRLAB_DOCS_TOKEN --repo sunerpy/bedrock-gateway-rust
```

Without it the workflow fails at its first step and says so, and nothing reaches FirLab until someone opens a sync pull
request there by hand.

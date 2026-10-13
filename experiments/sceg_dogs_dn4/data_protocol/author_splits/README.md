# SCEG author-provided Dogs and Cars class splits

These are the original text attachments the user obtained from the paper authors and supplied on 2026-10-13. The files were copied byte-for-byte; they list class assignments, not individual image paths. They are independent evidence and do not replace the frozen DN4 CSVs or configuration used for the current run.

| File | Meaning | Train / validation / test | SHA256 |
|---|---|---|---|
| [StanfordDogs_fewshot_splits.txt](StanfordDogs_fewshot_splits.txt) | Exact breed directory names including synset IDs | 70 / 20 / 30 | `350aa6cbc28605cfe5371354ec8f440878825fb420960e51fcc9766809c55df1` |
| [StanfordCars_fewshot_splits.txt](StanfordCars_fewshot_splits.txt) | Original category IDs, **1-based** | 130 / 17 / 49 | `2677107210b2bf1c9dfe1741ddc7d6d8dad992232ff04efa24ba4676aa48052c` |

Both lists are complete and mutually exclusive: Dogs covers the same 120 breed names as the existing Stanford archive; Cars covers IDs 1 through 196 exactly once. There is no Cars training run in the current task.

The current Dogs run uses the released [DN4 CSV protocol at commit 440857b23a5bc9798004f81f732c5cc9272a99b8](https://github.com/WenbinLee/DN4/tree/440857b23a5bc9798004f81f732c5cc9272a99b8/DN4_2019_Version/dataset/StanfordDogs). Comparing the complete CSV label sets and actual prepared ImageFolder directories with the author attachment shows **62 of 120 class assignments differ**. The matrix below uses DN4 splits as rows and author splits as columns:

| DN4 split | Author train | Author validation | Author test |
|---|---:|---:|---:|
| train | 45 | 11 | 14 |
| validation | 9 | 4 | 7 |
| test | 16 | 5 | 9 |

Full changed assignments and per-split differences are retained in [author_split_comparison.json](../author_split_comparison.json). In particular, 14 author test classes were trained in the DN4 run. Its checkpoint cannot become an author-split reproduction by changing only the test classes.

For reproducing this paper's class protocol, use the author-provided lists. Existing DN4 weights and results remain a baseline on the public DN4 protocol. An author-split experiment requires its own training and records; this attachment upload does not start it. The user has requested pausing the current DN4 work and disabling its automatic cleanup/commit workflow.

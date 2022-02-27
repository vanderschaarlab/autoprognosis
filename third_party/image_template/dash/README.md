---
title: Test4
emoji: ðŸ“ˆ
colorFrom: red
colorTo: red
sdk: streamlit
app_file: app.py
pinned: false
---

# Configuration

`title`: _string_
Display title for the Space

`emoji`: _string_
Space emoji (emoji-only character allowed)

`colorFrom`: _string_
Color for Thumbnail gradient (red, yellow, green, blue, indigo, purple, pink, gray)

`colorTo`: _string_
Color for Thumbnail gradient (red, yellow, green, blue, indigo, purple, pink, gray)

`sdk`: _string_
Can be either `gradio`, `streamlit`, or `static`

`sdk_version` : _string_
Only applicable for `streamlit` SDK.
See [doc](https://hf.co/docs/hub/spaces) for more info on supported versions.

`app_file`: _string_
Path to your main application file (which contains either `gradio` or `streamlit` Python code, or `static` html code).
Path is relative to the root of the repository.

`models`: _List[string]_
HF model IDs (like "gpt2" or "deepset/roberta-base-squad2") used in the Space.
Will be parsed automatically from your code if not specified here.

`datasets`: _List[string]_
HF dataset IDs (like "common_voice" or "oscar-corpus/OSCAR-2109") used in the Space.
Will be parsed automatically from your code if not specified here.

`pinned`: _boolean_
Whether the Space stays on top of your list.

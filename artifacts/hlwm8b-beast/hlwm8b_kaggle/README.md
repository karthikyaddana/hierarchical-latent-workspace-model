# HLWM 8B Kaggle code

Attach this code bundle and a frozen `hlwm-beast-teacher-snapshot.zip` to the generated
notebook. Use T4 x2 and Save & Run All. The notebook preflights the pinned Qwen model,
resumes exact optimizer/RNG state, stops before Kaggle's session boundary, evaluates
against untouched anchors, and emits split checkpoint downloads.

"""Quick smoke-test for the ML pipeline."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.ml_model import run_full_pipeline, load_model, predict_single

print("=" * 55)
print("  BenefiAI - ML Pipeline Smoke Test")
print("=" * 55)

# 1. Run full pipeline
result = run_full_pipeline()
assert result["success"], f"Pipeline failed: {result.get('error')}"
print("\n[PASS] Full pipeline completed successfully")

# 2. Check cleaning log
cl = result["cleaning_log"]
print(f"\n  Cleaning log:")
print(f"    Duplicates removed : {cl['duplicates_removed']}")
print(f"    Nulls removed      : {cl['nulls_removed']}")
print(f"    Clean records used : {cl['final_rows']}")

# 3. Check metrics
m = result["metrics"]
print(f"\n  Performance metrics:")
print(f"    Train accuracy : {m['train_accuracy']*100:.2f}%")
print(f"    Test accuracy  : {m['accuracy']*100:.2f}%")
print(f"    Precision      : {m['precision']*100:.2f}%")
print(f"    Recall         : {m['recall']*100:.2f}%")
print(f"    F1 Score       : {m['f1_score']*100:.2f}%")
print(f"    ROC-AUC        : {m['roc_auc']:.4f}")
print(f"    5-Fold CV      : {m['cv_mean_accuracy']*100:.2f}% +/- {m['cv_std']*100:.2f}%")
print(f"    Confusion Mat  : {m['confusion_matrix']}")
print(f"    Train rows     : {m['train_size']}")
print(f"    Test rows      : {m['test_size']}")

# 4. Reload from disk and do a single prediction
bundle = load_model()
assert bundle is not None, "Model not saved to disk!"
print("\n[PASS] Model loaded from disk")

sample = {
    "age":               35,
    "family_income":     80000.0,
    "family_members":    5,
    "employment_status": "Unemployed",
    "education_level":   "Primary",
    "disability_status": "No",
}
pred = predict_single(bundle, sample)
print(f"\n  Single prediction:")
print(f"    Applicant      : Age {sample['age']}, Income Rs.{sample['family_income']:,.0f}")
print(f"    Employment     : {sample['employment_status']}")
print(f"    Education      : {sample['education_level']}")
print(f"    Prediction     : {pred['label']}")
print(f"    Confidence     : {pred['confidence']*100:.1f}%")
print(f"    Flags fired    : {pred['flags']['total_flags']}/4")
print(f"    Per-capita Inc : Rs.{pred['per_capita_income']:,.0f}")
print("\n[PASS] Single prediction successful")
print("=" * 55)
print("  ALL TESTS PASSED")
print("=" * 55)

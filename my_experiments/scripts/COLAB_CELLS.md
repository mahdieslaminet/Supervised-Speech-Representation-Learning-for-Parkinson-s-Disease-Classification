# اجرای مقاوم روی Colab (یک بار آماده‌سازی + چهار سلول)

## یک بار، روی کامپیوتر خودت (از ریشه‌ی ریپو)
اسکریپت‌های جدید (`run_fold.py`, `run_all.py`, `check_folds.py`, `collect_results.py`, `apply_patches.py`,
`pack_data.py`, `colab_bootstrap.py`) را در `my_experiments/scripts/` بگذار، بعد:

    python my_experiments/scripts/pack_data.py --data_dir preprocess/pcgita_full_100spk

پوشه‌ی ساخته‌شده `colab_pack` را روی Google Drive آپلود کن و نامش را بگذار `idiap_pd`
(یعنی `MyDrive/idiap_pd/data/...tar` و `MyDrive/idiap_pd/repo_code/`).
یک فایل tar خیلی سریع‌تر از هزاران فایل .npy آپلود می‌شود.

## در Colab (بعد از هر قطع شدن، همین سلول‌ها را دوباره اجرا کن)
Runtime > Change runtime type > GPU

سلول ۱:
    from google.colab import drive
    drive.mount('/content/drive')

سلول ۲ (داده را روی دیسک محلی کپی/باز می‌کند، results را به Drive وصل می‌کند، پچ‌ها را می‌زند):
    !python ./my_experiments/scripts/colab_bootstrap.py
    %cd /content/repo

سلول ۳ (بررسی سلامت فولدها):
    !python my_experiments/scripts/check_folds.py preprocess/pcgita_full_100spk

سلول ۴ (اجرا؛ هر بار که قطع شد همین را دوباره بزن، از همان‌جا ادامه می‌دهد):
    !python my_experiments/scripts/run_all.py --methods baseline --seeds 0 --folds 1-10

جمع‌بندی:
    !python my_experiments/scripts/collect_results.py

<div dir="rtl">

# طبقه‌بندی بیماری پارکینسون از روی گفتار: بازتولید یادگیری بازنمایی نظارت‌شده (PC-GITA)

بازتولید و اعتبارسنجی مقاله **«Supervised Speech Representation Learning for Parkinson's Disease Classification»** (جان‌بخشی و کدراسی، مؤسسه Idiap) روی گوگل کولب. کد بر پایه مخزن رسمی نویسندگان ([idiap/pddetection-reps-learning](https://github.com/idiap/pddetection-reps-learning)) است و روی کل دیتاست PC-GITA اجرا شده است.

## این پروژه چه می‌کند؟

یک اتوانکودر CNN، نمایش فشرده‌ای از طیف‌نگار Mel گفتار یاد می‌گیرد. چهار حالت آموزش مقایسه می‌شوند:

| روش | توضیح |
|---|---|
| `baseline` | اتوانکودر بدون نظارت (بدون وظیفه کمکی) |
| `adv` | شاخه خصمانه تشخیص گوینده (نمایش مستقل از گوینده) |
| `pd` | طبقه‌بند پارکینسون همراه با اتوانکودر (نمایش متمایزکننده بیماری) |
| `fusion` | ترکیب هر دو وظیفه کمکی |

سپس از نمایش لایه گلوگاه (bottleneck) برای تشخیص بیمار پارکینسونی از فرد سالم استفاده می‌شود. نتیجه نهایی هر گوینده با رأی‌گیری نرم (Soft Voting) روی همه قطعات او به دست می‌آید.

## تنظیمات (مطابق مقاله)

- داده: PC-GITA، ۱۰۰ گوینده (۵۰ بیمار و ۵۰ سالم)، زبان اسپانیایی
- ورودی: قطعات ۵۰۰ میلی‌ثانیه‌ای طیف Mel با ۵۰٪ هم‌پوشانی و ۱۲۶ باند
- ارزیابی: اعتبارسنجی متقاطع ۱۰ فولدی، طبقه‌ای و مستقل از گوینده
- معیارها: دقت و AUC (تشخیص پارکینسون)، دقت و AUC (تشخیص هویت گوینده)
- ابرپارامترها: lambda = alpha = 0.01
- Seed: فقط seed 0 (مقاله میانگین ۵ seed را گزارش کرده است)

## ساختار مخزن

```
repo_code/
  my_experiments/
    scripts/            # پیش‌پردازش، run_fold.py، run_all.py، check_folds.py،
                        # colab_bootstrap.py، apply_patches.py، make_report.py
  upstream_auxiliary/   # کد مخزن اصلی Idiap (وصله‌شده)
notebooks/
  Session_*.ipynb       # اجرای آزمایش‌ها روی کولب
  Validation_Notebook.ipynb   # بررسی‌های اعتبارسنجی (فولدها، Soft Voting، ROC، بازسازی طیف)
preprocess/             # به‌صورت محلی ساخته می‌شود (در مخزن نیست)
results/                # برای هر روش/seed/فولد ساخته می‌شود (در مخزن نیست)
```

## نحوه اجرا

1. دیتاست PC-GITA را از نویسندگان آن بگیرید (به دلیل مجوز، در این مخزن **نیست**).
2. آن را در مسیر `preprocess/pcgita_full_100spk` پیش‌پردازش کنید (۱۰ فولد).
3. یک نوت‌بوک را در گوگل کولب باز کنید و گوگل درایو را متصل کنید.
4. محیط را آماده کنید و وصله‌ها را اعمال کنید:
   ```bash
   python my_experiments/scripts/colab_bootstrap.py
   python my_experiments/scripts/apply_patches.py /content/repo
   ```
5. سلامت فولدها را بررسی کنید و آموزش را اجرا کنید:
   ```bash
   python my_experiments/scripts/check_folds.py preprocess/pcgita_full_100spk
   python my_experiments/scripts/run_all.py --methods baseline adv pd fusion --seeds 0 --folds 1-10
   ```
6. برای بازمحاسبه معیارها از فایل‌های `output_K.npy` و رسم نمودارها، `Validation_Notebook.ipynb` را اجرا کنید.

## نتایج

| روش | دقت (٪) | AUC |
|---|---|---|
| Baseline | _پر شود_ | _پر شود_ |
| Adversarial | _پر شود_ | _پر شود_ |
| PD discriminative | _پر شود_ | _پر شود_ |
| Fusion | _پر شود_ | _پر شود_ |

مقادیر گزارش‌شده در مقاله (میانگین ۵ seed): Baseline ‏۶۶٫۲٪ / ۰٫۷۷، Adversarial ‏۷۲٫۰٪ / ۰٫۸۴، PD ‏۷۱٫۰٪ / ۰٫۷۸، Fusion ‏۷۵٫۴٪ / ۰٫۸۰.

## نکات

- چون فقط یک seed اجرا شده، نتایج ممکن است کمی با مقاله فرق کنند.
- دیتاست بازنشر نمی‌شود. لطفاً دسترسی را از نویسندگان PC-GITA درخواست کنید.

## منابع و ارجاع

- مقاله: P. Janbakhshi و I. Kodrasi، *Supervised Speech Representation Learning for Parkinson's Disease Classification*، مؤسسه تحقیقاتی Idiap (arXiv:2106.00531).
- کد اصلی: https://github.com/idiap/pddetection-reps-learning
- دیتاست: J. R. Orozco-Arroyave و همکاران، PC-GITA، LREC 2014.

## مجوز

مجوز خود را اینجا بنویسید. به مجوز مخزن اصلی و دیتاست PC-GITA احترام بگذارید.

</div>

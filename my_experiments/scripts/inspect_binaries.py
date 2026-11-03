# -*- coding: utf-8 -*-
"""
اسکریپت استخراج و تبدیل فایل‌های باینری پروژه (.npy و .ckpt) به فرمت‌های خوانا (.txt و .csv)
در یک پوشه مجزا (readable_exports) برای ارائه و بررسی راحت در ویندوز (Notepad / Excel).
این اسکریپت فایل‌های اصلی را فقط در حالت Read-Only می‌خواند و هیچ تغییری در آن‌ها ایجاد نمی‌کند.

نحوه اجرا از ریشه مخزن (Repo Root):
    python my_experiments/scripts/inspect_binaries.py
"""

import argparse
import glob
import json
import os
import pprint
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def inspect_feature_npy(npy_path: Path, out_dir: Path):
    """بررسی فایل باینری ویژگی صوتی (.npy) و ذخیره اطلاعات آماری و پیش‌نمایش"""
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = np.load(npy_path, allow_pickle=True)
    base_name = npy_path.stem

    info_file = out_dir / f"{base_name}_info.txt"
    csv_file = out_dir / f"{base_name}_preview_matrix.csv"

    with open(info_file, "w", encoding="utf-8") as f:
        f.write("=================================================================\n")
        f.write("       گزارش شناسنامه فایل باینری ویژگی صوتی (Acoustic Feature)\n")
        f.write("=================================================================\n")
        f.write(f"مسیر فایل اصلی : {npy_path.as_posix()}\n")
        f.write(f"نوع داده (Dtype): {arr.dtype}\n")
        f.write(f"ابعاد ماتریس (Shape): {arr.shape}  --> (تعداد کانال‌های فرکانسی × تعداد فریم‌های زمانی)\n")
        if np.issubdtype(arr.dtype, np.number):
            f.write(f"کمترین مقدار (Min) : {arr.min():.6f}\n")
            f.write(f"بیشترین مقدار (Max): {arr.max():.6f}\n")
            f.write(f"میانگین (Mean)     : {arr.mean():.6f}\n")
            f.write(f"انحراف معیار (Std)  : {arr.std():.6f}\n")
        f.write("\n-----------------------------------------------------------------\n")
        f.write("پیش‌نمایش مقادیر ماتریس (۵ کانال فرکانسی اول در ۸ فریم زمانی اول):\n")
        f.write("-----------------------------------------------------------------\n")
        if arr.ndim >= 2:
            f.write(np.array2string(arr[:5, :8], precision=4, separator=", ") + "\n")
        else:
            f.write(np.array2string(arr[:10], precision=4, separator=", ") + "\n")

    if arr.ndim == 2:
        max_frames = min(50, arr.shape[1])
        df = pd.DataFrame(
            arr[:, :max_frames],
            index=[f"FreqBin_{i}" for i in range(arr.shape[0])],
            columns=[f"Frame_{j}" for j in range(max_frames)],
        )
        df.to_csv(csv_file, index_label="Frequency_Channel")

    print(f"[OK] Feature NPY exported -> {info_file.as_posix()}")


def inspect_output_npy(npy_path: Path, out_dir: Path):
    """بررسی فایل باینری خروجی ارزیابی مدل"""
    out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(npy_path, allow_pickle=True)
    base_name = npy_path.stem

    txt_file = out_dir / f"{base_name}_summary.txt"
    obj = data.item() if (isinstance(data, np.ndarray) and data.shape == () and data.dtype == object) else data

    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("=================================================================\n")
        f.write("       گزارش محتوای فایل خروجی ارزیابی مدل (Model Output NPY)\n")
        f.write("=================================================================\n")
        f.write(f"مسیر فایل اصلی : {npy_path.as_posix()}\n")
        f.write(f"نوع ساختار داده: {type(obj).__name__}\n")
        if isinstance(data, np.ndarray):
            f.write(f"ابعاد آرایه نامپی: {data.shape} | Dtype: {data.dtype}\n")
        f.write("=================================================================\n\n")

        if isinstance(obj, dict):
            f.write(f"کلیدهای موجود در فایل خروجی: {list(obj.keys())}\n\n")
            for k, v in obj.items():
                f.write(f"---> کلید [{k}]:\n")
                if isinstance(v, np.ndarray):
                    f.write(f"  نوع: آرایه نامپی | ابعاد (Shape): {v.shape} | Dtype: {v.dtype}\n")
                    f.write(pprint.pformat(v, indent=4, width=100) + "\n\n")
                    if v.ndim <= 2 and v.size > 0:
                        pd.DataFrame(v).to_csv(out_dir / f"{base_name}_{k}.csv", index=False)
                else:
                    f.write(pprint.pformat(v, indent=4, width=100) + "\n\n")
        elif isinstance(obj, np.ndarray):
            f.write("محتوای آرایه:\n")
            f.write(pprint.pformat(obj, indent=2, width=100) + "\n")
            if obj.dtype == object:
                for idx, elem in enumerate(obj.flat):
                    f.write(f"\n--- عنصر شماره [{idx}] (نوع: {type(elem).__name__}) ---\n")
                    f.write(pprint.pformat(elem, indent=4, width=100) + "\n")
                    if isinstance(elem, np.ndarray) and elem.ndim <= 2:
                        pd.DataFrame(elem).to_csv(out_dir / f"{base_name}_part{idx}.csv", index=False)
            elif obj.ndim <= 2:
                pd.DataFrame(obj).to_csv(out_dir / f"{base_name}_table.csv", index=False)
        else:
            f.write(pprint.pformat(obj, indent=2, width=100) + "\n")

    print(f"[OK] Model Output NPY exported -> {txt_file.as_posix()}")


def inspect_ckpt(ckpt_path: Path, out_dir: Path):
    """کالبدشکافی فایل چک‌پوینت پای‌تورچ (.ckpt)"""
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    base_name = ckpt_path.stem

    txt_file = out_dir / f"{base_name}_structure.txt"
    csv_file = out_dir / f"{base_name}_layers_weights.csv"
    layer_rows = []

    with open(txt_file, "w", encoding="utf-8") as f:
        f.write("=================================================================\n")
        f.write("       شناسنامه و ساختار فایل چک‌پوینت مدل (PyTorch Checkpoint)\n")
        f.write("=================================================================\n")
        f.write(f"مسیر فایل اصلی : {ckpt_path.as_posix()}\n")
        if isinstance(ckpt, dict):
            f.write(f"کلیدهای اصلی ذخیره‌شده در چک‌پوینت: {list(ckpt.keys())}\n")
        f.write("=================================================================\n\n")

        if isinstance(ckpt, dict):
            for section_key, section_val in ckpt.items():
                f.write("-----------------------------------------------------------------\n")
                f.write(f"بخش [{section_key}]:\n")
                f.write("-----------------------------------------------------------------\n")

                if isinstance(section_val, dict) and section_val:
                    first_val = next(iter(section_val.values()))
                    if torch.is_tensor(first_val):
                        total_params = 0
                        f.write("  [لایه‌های شبکه عصبی و ابعاد تانسورهای وزن (State Dict)]:\n")
                        for layer_name, tensor in section_val.items():
                            num_p = tensor.numel()
                            total_params += num_p
                            shape_str = str(list(tensor.shape))
                            f.write(
                                f"    - {layer_name:42s} | Shape: {shape_str:18s} | "
                                f"Params: {num_p:10,d} | Dtype: {tensor.dtype}\n"
                            )
                            t_float = tensor.detach().float()
                            layer_rows.append({
                                "Section": section_key,
                                "Layer_Name": layer_name,
                                "Shape": shape_str,
                                "Num_Parameters": num_p,
                                "Dtype": str(tensor.dtype),
                                "Weight_Min": float(t_float.min().item()) if num_p > 0 else 0.0,
                                "Weight_Max": float(t_float.max().item()) if num_p > 0 else 0.0,
                                "Weight_Mean": float(t_float.mean().item()) if num_p > 0 else 0.0,
                                "Weight_Std": float(t_float.std().item()) if num_p > 1 else 0.0,
                            })
                        f.write(f"\n  *** مجموع تعداد پارامترهای بخش [{section_key}]: {total_params:,} پارامتر ***\n\n")
                    elif "param_groups" in section_val:
                        f.write("  [وضعیت بهینه‌ساز (Optimizer State)]:\n")
                        f.write("  گروه‌های پارامتری و نرخ یادگیری (param_groups):\n")
                        for pg in section_val["param_groups"]:
                            pg_clean = {k: v for k, v in pg.items() if k != "params"}
                            pg_clean["num_tensors_in_group"] = len(pg.get("params", []))
                            f.write(f"    {json.dumps(pg_clean, ensure_ascii=False)}\n")
                        f.write("\n")
                    else:
                        f.write("  [تنظیمات و هایپرپارامترهای ذخیره‌شده (Config Dict)]:\n")
                        f.write(json.dumps(section_val, indent=2, ensure_ascii=False, default=str) + "\n\n")
                else:
                    f.write(f"  مقدار: {section_val}\n\n")

    if layer_rows:
        pd.DataFrame(layer_rows).to_csv(csv_file, index=False)

    print(f"[OK] Checkpoint CKPT exported -> {txt_file.as_posix()}")


def main():
    ap = argparse.ArgumentParser(description="Export readable summaries of binary .npy and .ckpt files into a separate folder.")
    ap.add_argument("--repo", default=".", help="مسیر ریشه مخزن پروژه")
    ap.add_argument("--out_dir", default="readable_exports", help="نام پوشه مجزا برای ذخیره خروجی‌های خوانا")
    ap.add_argument("--max_features", type=int, default=2, help="حداکثر تعداد نمونه فایل ویژگی صوتی برای استخراج")
    args = ap.parse_args()

    repo = Path(args.repo).resolve()
    export_root = repo / args.out_dir
    export_root.mkdir(parents=True, exist_ok=True)

    print(f"=== شروع استخراج فایل‌های باینری در پوشه مجزا: {export_root.as_posix()} ===\n")

    feat_files = sorted(repo.glob("preprocess/*/features_data/*/*.npy"))
    if feat_files:
        feat_out_dir = export_root / "1_acoustic_features_samples"
        for fpath in feat_files[: args.max_features]:
            inspect_feature_npy(fpath, feat_out_dir)
    else:
        print("[INFO] هیچ فایل ویژگی صوتی در مسیر preprocess/*/features_data/*/*.npy یافت نشد.")

    out_files = sorted(repo.glob("results/*/ds-*/outputs/output_*.npy"))
    if out_files:
        for fpath in out_files:
            exp_name = fpath.parents[2].name
            out_dir = export_root / "2_model_evaluation_outputs" / exp_name
            inspect_output_npy(fpath, out_dir)
    else:
        print("[INFO] هیچ فایل خروجی مدلی در مسیر results/*/ds-*/outputs/output_*.npy یافت نشد.")

    ckpt_files = sorted(repo.glob("results/*/*/outputs/saved_model/*.ckpt"))
    if ckpt_files:
        for cpath in ckpt_files:
            exp_name = cpath.parents[3].name
            stage_folder = cpath.parents[2].name
            out_dir = export_root / "3_model_checkpoints" / exp_name / stage_folder
            inspect_ckpt(cpath, out_dir)
    else:
        print("[INFO] هیچ فایل چک‌پوینتی در مسیر results/*/*/outputs/saved_model/*.ckpt یافت نشد.")

    print(f"\n=== پایان عملیات! تمامی فایل‌های متنی (.txt) و جدولی (.csv) در پوشه '{export_root.name}' آماده شدند. ===")


if __name__ == "__main__":
    main()
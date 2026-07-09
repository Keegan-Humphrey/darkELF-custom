#!/usr/bin/env python3
"""Crop a plot PDF horizontally so its plotting borders align with reference PDFs.

The default left crop (3.30 pt) is calibrated for money_eD.pdf relative to
money_disk.pdf / money_halo.pdf. The output remains vector PDF.

Requires: pip install pymupdf
"""

from __future__ import annotations

import argparse
from pathlib import Path
import fitz  # PyMuPDF


def align_pdf(
    source: str | Path,
    reference: str | Path,
    output: str | Path,
    *,
    left_crop_pt: float = 3.30,
) -> None:
    """Match source page width to reference by cropping source without scaling.

    `left_crop_pt` controls the horizontal translation. The remaining required
    crop is automatically taken from the right edge.
    """
    source = Path(source)
    reference = Path(reference)
    output = Path(output)

    with fitz.open(source) as src, fitz.open(reference) as ref:
        if len(src) != 1 or len(ref) < 1:
            raise ValueError("This helper expects a one-page source PDF.")

        src_page = src[0]
        target_width = ref[0].rect.width
        source_width = src_page.rect.width
        source_height = src_page.rect.height

        total_crop = source_width - target_width
        right_crop_pt = total_crop - left_crop_pt

        if total_crop < 0:
            raise ValueError(
                f"Source is {abs(total_crop):.3f} pt narrower than reference; "
                "cropping alone cannot match it."
            )
        if left_crop_pt < 0 or right_crop_pt < 0:
            raise ValueError(
                f"Invalid crop: left={left_crop_pt:.3f} pt, "
                f"right={right_crop_pt:.3f} pt."
            )

        clip = fitz.Rect(
            left_crop_pt,
            0,
            left_crop_pt + target_width,
            source_height,
        )

        out = fitz.open()
        out_page = out.new_page(width=target_width, height=source_height)
        out_page.show_pdf_page(
            out_page.rect,
            src,
            0,
            clip=clip,
            keep_proportion=False,
        )
        out.save(output, garbage=4, deflate=True)
        out.close()

    print(f"Saved: {output}")
    print(f"Left crop:  {left_crop_pt:.3f} pt")
    print(f"Right crop: {right_crop_pt:.3f} pt")
    print(f"Final width: {target_width:.3f} pt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="PDF to crop, e.g. money_eD.pdf")
    parser.add_argument("reference", help="Width reference, e.g. money_disk.pdf")
    parser.add_argument("output", help="Output PDF")
    parser.add_argument(
        "--left-crop",
        type=float,
        default=3.30,
        help="Crop from left in PDF points (default: 3.30)",
    )
    args = parser.parse_args()

    align_pdf(
        args.source,
        args.reference,
        args.output,
        left_crop_pt=args.left_crop,
    )


if __name__ == "__main__":
    main()

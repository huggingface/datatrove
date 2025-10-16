#!/usr/bin/env python3
"""
Example 08b: PDF Threshold Analysis

Analyzes PDF classification thresholds for manual evaluation and tuning.

Components:
- PDFWarcReader: Read PDFs from WARC files
- PDFScannedPredictor: Classify PDFs by OCR probability
- Matplotlib/Seaborn: Generate threshold visualizations

Usage:
    python spec/phase3/examples/08b_pdf_threshold_analysis.py
"""

import io
import os
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from datatrove.pipeline.media.predictor.scanned_pdf_predictor import PDFScannedPredictor
from datatrove.pipeline.readers.pdf_warc import PDFWarcReader
from datatrove.utils.logging import logger


class PDFThresholdAnalyzer:
    def __init__(self, data_folder="spec/phase3/data", model_path="spec/phase3/data/pdf_classifier_real_data.xgb"):
        self.data_folder = data_folder
        self.model_path = model_path
        self.output_dir = "spec/phase3/threshold_analysis"

        # Create output directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(f"{self.output_dir}/samples", exist_ok=True)
        os.makedirs(f"{self.output_dir}/plots", exist_ok=True)

        # Define thresholds
        self.thresholds = [
            (0.0, 0.2, "very_low_ocr"),
            (0.2, 0.4, "low_ocr"),
            (0.4, 0.6, "medium_ocr"),
            (0.6, 0.8, "high_ocr"),
            (0.8, 1.0, "very_high_ocr")
        ]

        # Storage for analysis
        self.pdf_classifications = []
        self.threshold_samples = {label: [] for _, _, label in self.thresholds}
        self.samples_per_threshold = 5

        # Initialize predictor
        logger.info(f"Loading model from {self.model_path}")
        self.predictor = PDFScannedPredictor(
            path_to_model=self.model_path,
            num_pages_to_sample=8
        )

    def process_all_warcs(self):
        """Process all WARC files and classify PDFs."""
        warc_files = [f for f in os.listdir(self.data_folder) if f.endswith('.warc.gz')]
        logger.info(f"Found {len(warc_files)} WARC files: {warc_files}")

        total_pdfs = 0
        valid_pdfs = 0
        successful_classifications = 0
        start_time = time.time()

        for warc_file in warc_files:
            logger.info(f"\nProcessing {warc_file}...")
            file_start = time.time()

            reader = PDFWarcReader(
                data_folder=self.data_folder,
                glob_pattern=warc_file,
                limit=-1
            )

            file_pdfs = 0
            file_successful = 0

            for doc in reader.run():
                total_pdfs += 1
                file_pdfs += 1

                # Skip non-PDF files
                if not doc.text.startswith(b'%PDF-'):
                    continue

                valid_pdfs += 1

                # Classify PDF
                try:
                    result = self.predictor.predict(doc.text)

                    if "prediction_failed" not in result:
                        ocr_prob = result["ocr_prob"]

                        # Store classification data
                        classification = {
                            'id': doc.id,
                            'warc_file': warc_file,
                            'url': doc.metadata.get('url', ''),
                            'content_length': doc.metadata.get('content_length', 0),
                            'is_truncated': doc.metadata.get('is_truncated', False),
                            'ocr_prob': ocr_prob,
                            'is_form': result.get('is_form', False),
                            'garbled_text_ratio': result.get('garbled_text_ratio', 0.0),
                            'num_pages': result.get('num_pages', 0),
                            'is_encrypted': result.get('is_encrypted', False),
                            'pdf_bytes': doc.text  # Store for sampling
                        }

                        self.pdf_classifications.append(classification)

                        # Check if this PDF should be sampled for any threshold
                        self._check_for_sampling(classification)

                        successful_classifications += 1
                        file_successful += 1

                except Exception as e:
                    continue

                # Progress update
                if file_pdfs % 100 == 0:
                    elapsed = time.time() - file_start
                    rate = file_pdfs / elapsed if elapsed > 0 else 0
                    logger.info(f"  {file_pdfs} PDFs processed ({rate:.1f}/sec), {file_successful} classified")

            file_time = time.time() - file_start
            logger.info(f"  Completed {warc_file}: {file_pdfs} PDFs, {file_successful} classified in {file_time:.1f}s")

        total_time = time.time() - start_time
        logger.info(f"\nProcessing complete:")
        logger.info(f"  Total PDFs: {total_pdfs}")
        logger.info(f"  Valid PDFs: {valid_pdfs}")
        logger.info(f"  Successful classifications: {successful_classifications}")
        logger.info(f"  Success rate: {successful_classifications/max(1, valid_pdfs):.1%}")
        logger.info(f"  Total time: {total_time:.1f}s")

    def _check_for_sampling(self, classification):
        """Check if this PDF should be sampled for any threshold."""
        ocr_prob = classification['ocr_prob']

        for min_thresh, max_thresh, label in self.thresholds:
            if min_thresh <= ocr_prob < max_thresh:
                if len(self.threshold_samples[label]) < self.samples_per_threshold:
                    self.threshold_samples[label].append(classification)
                    logger.info(f"    Sampled PDF {classification['id']} for {label} threshold ({ocr_prob:.3f})")
                break

    def save_sample_pdfs(self):
        """Save sample PDFs to disk for manual evaluation."""
        logger.info(f"\nSaving sample PDFs...")

        for threshold_label, samples in self.threshold_samples.items():
            if not samples:
                continue

            threshold_dir = f"{self.output_dir}/samples/{threshold_label}"
            os.makedirs(threshold_dir, exist_ok=True)

            # Save sample info
            sample_info = []

            for i, sample in enumerate(samples):
                filename = f"{threshold_label}_{i+1:02d}_{sample['id']}.pdf"
                filepath = os.path.join(threshold_dir, filename)

                # Save PDF bytes
                with open(filepath, 'wb') as f:
                    f.write(sample['pdf_bytes'])

                # Record sample info (without pdf_bytes for JSON serialization)
                info = {k: v for k, v in sample.items() if k != 'pdf_bytes'}
                info['saved_filename'] = filename
                sample_info.append(info)

                logger.info(f"  Saved {filepath} (OCR prob: {sample['ocr_prob']:.3f})")

            # Save sample metadata
            import json
            with open(f"{threshold_dir}/sample_info.json", 'w') as f:
                json.dump(sample_info, f, indent=2)

    def generate_statistics(self):
        """Generate comprehensive statistics."""
        if not self.pdf_classifications:
            logger.info("No classifications to analyze")
            return

        logger.info(f"\nGenerating statistics from {len(self.pdf_classifications)} classified PDFs...")

        df = pd.DataFrame([{k: v for k, v in pdf.items() if k != 'pdf_bytes'}
                          for pdf in self.pdf_classifications])

        # Basic statistics
        stats = {
            'total_classified': len(df),
            'ocr_prob_stats': {
                'mean': df['ocr_prob'].mean(),
                'median': df['ocr_prob'].median(),
                'std': df['ocr_prob'].std(),
                'min': df['ocr_prob'].min(),
                'max': df['ocr_prob'].max(),
                'q25': df['ocr_prob'].quantile(0.25),
                'q75': df['ocr_prob'].quantile(0.75)
            },
            'threshold_distribution': {},
            'characteristics_by_threshold': {}
        }

        # Threshold distribution
        for min_thresh, max_thresh, label in self.thresholds:
            mask = (df['ocr_prob'] >= min_thresh) & (df['ocr_prob'] < max_thresh)
            count = mask.sum()
            percentage = count / len(df) * 100

            stats['threshold_distribution'][label] = {
                'count': int(count),
                'percentage': percentage,
                'range': f"{min_thresh}-{max_thresh}"
            }

            # Characteristics within this threshold
            if count > 0:
                subset = df[mask]
                stats['characteristics_by_threshold'][label] = {
                    'avg_pages': subset['num_pages'].mean(),
                    'form_percentage': subset['is_form'].mean() * 100,
                    'avg_garbled_ratio': subset['garbled_text_ratio'].mean(),
                    'truncated_percentage': subset['is_truncated'].mean() * 100,
                    'encrypted_percentage': subset['is_encrypted'].mean() * 100,
                    'avg_content_length': subset['content_length'].mean()
                }

        # Save statistics
        import json
        with open(f"{self.output_dir}/statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)

        # Print summary
        logger.info(f"\nStatistics Summary:")
        logger.info(f"  Total classified PDFs: {stats['total_classified']}")
        logger.info(f"  OCR probability - Mean: {stats['ocr_prob_stats']['mean']:.3f}, Median: {stats['ocr_prob_stats']['median']:.3f}")
        logger.info(f"\nThreshold Distribution:")
        for label, data in stats['threshold_distribution'].items():
            logger.info(f"  {label} ({data['range']}): {data['count']} PDFs ({data['percentage']:.1f}%)")

        return df, stats

    def create_visualizations(self, df, stats):
        """Create plots and tables for analysis."""
        logger.info(f"\nCreating visualizations...")

        # Set style
        plt.style.use('default')
        sns.set_palette("husl")

        # 1. OCR Probability Distribution
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.hist(df['ocr_prob'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('OCR Probability')
        plt.ylabel('Count')
        plt.title('Distribution of OCR Probabilities')
        plt.grid(True, alpha=0.3)

        # 2. Threshold Distribution Bar Chart
        plt.subplot(2, 2, 2)
        threshold_labels = [label for _, _, label in self.thresholds]
        threshold_counts = [stats['threshold_distribution'][label]['count'] for label in threshold_labels]
        threshold_ranges = [stats['threshold_distribution'][label]['range'] for label in threshold_labels]

        bars = plt.bar(range(len(threshold_labels)), threshold_counts, alpha=0.7)
        plt.xlabel('OCR Probability Threshold')
        plt.ylabel('Number of PDFs')
        plt.title('PDF Distribution by Threshold')
        plt.xticks(range(len(threshold_labels)), [f"{label}\n({r})" for label, r in zip(threshold_labels, threshold_ranges)], rotation=45)
        plt.grid(True, alpha=0.3)

        # Add count labels on bars
        for bar, count in zip(bars, threshold_counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(threshold_counts)*0.01,
                    str(count), ha='center', va='bottom')

        # 3. Box plot of OCR probabilities by characteristics
        plt.subplot(2, 2, 3)

        # Create categories for box plot
        ocr_categories = []
        for _, row in df.iterrows():
            if row['ocr_prob'] < 0.1:
                ocr_categories.append('Very Low\n(<0.1)')
            elif row['ocr_prob'] < 0.5:
                ocr_categories.append('Low-Medium\n(0.1-0.5)')
            else:
                ocr_categories.append('High\n(≥0.5)')

        df_plot = df.copy()
        df_plot['ocr_category'] = ocr_categories

        sns.boxplot(data=df_plot, x='ocr_category', y='num_pages')
        plt.title('Page Count by OCR Probability Category')
        plt.ylabel('Number of Pages')

        # 4. Scatter plot: OCR prob vs Garbled ratio
        plt.subplot(2, 2, 4)
        scatter = plt.scatter(df['garbled_text_ratio'], df['ocr_prob'], alpha=0.6, c=df['num_pages'], cmap='viridis')
        plt.xlabel('Garbled Text Ratio')
        plt.ylabel('OCR Probability')
        plt.title('OCR Probability vs Garbled Text Ratio')
        plt.colorbar(scatter, label='Number of Pages')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/plots/ocr_probability_analysis.png", dpi=300, bbox_inches='tight')
        plt.show()

        # 5. Detailed characteristics table
        self._create_characteristics_table(stats)

    def _create_characteristics_table(self, stats):
        """Create a detailed characteristics comparison table."""

        # Prepare data for table
        table_data = []
        for label, char_stats in stats['characteristics_by_threshold'].items():
            thresh_info = stats['threshold_distribution'][label]

            row = {
                'Threshold': thresh_info['range'],
                'Category': label.replace('_', ' ').title(),
                'Count': thresh_info['count'],
                'Percentage': f"{thresh_info['percentage']:.1f}%",
                'Avg Pages': f"{char_stats['avg_pages']:.1f}",
                'Form %': f"{char_stats['form_percentage']:.1f}%",
                'Garbled Ratio': f"{char_stats['avg_garbled_ratio']:.3f}",
                'Truncated %': f"{char_stats['truncated_percentage']:.1f}%",
                'Encrypted %': f"{char_stats['encrypted_percentage']:.1f}%",
                'Avg Size (KB)': f"{char_stats['avg_content_length']/1024:.1f}"
            }
            table_data.append(row)

        # Create table plot
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.axis('tight')
        ax.axis('off')

        table_df = pd.DataFrame(table_data)
        table = ax.table(cellText=table_df.values, colLabels=table_df.columns,
                        cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)

        # Style the table
        for i in range(len(table_df.columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')

        plt.title('PDF Characteristics by OCR Probability Threshold', pad=20, fontsize=14, weight='bold')
        plt.savefig(f"{self.output_dir}/plots/characteristics_table.png", dpi=300, bbox_inches='tight')
        plt.show()

        # Save as CSV too
        table_df.to_csv(f"{self.output_dir}/characteristics_table.csv", index=False)
        logger.info(f"  Characteristics table saved to characteristics_table.csv")

    def generate_sampling_report(self):
        """Generate a report on the sampling results."""
        logger.info(f"\nSampling Report:")

        for min_thresh, max_thresh, label in self.thresholds:
            samples = self.threshold_samples[label]
            logger.info(f"\n{label.replace('_', ' ').title()} ({min_thresh}-{max_thresh}):")
            logger.info(f"  Samples collected: {len(samples)}/{self.samples_per_threshold}")

            if samples:
                ocr_probs = [s['ocr_prob'] for s in samples]
                logger.info(f"  OCR probability range: {min(ocr_probs):.3f} - {max(ocr_probs):.3f}")
                logger.info(f"  Sample PDFs:")
                for i, sample in enumerate(samples):
                    logger.info(f"    {i+1}. {sample['id']} (OCR: {sample['ocr_prob']:.3f}, Pages: {sample['num_pages']}, Form: {sample['is_form']})")

        # Create summary file
        summary = {
            'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_classifications': len(self.pdf_classifications),
            'samples_per_threshold': self.samples_per_threshold,
            'sampling_summary': {}
        }

        for min_thresh, max_thresh, label in self.thresholds:
            samples = self.threshold_samples[label]
            summary['sampling_summary'][label] = {
                'threshold_range': f"{min_thresh}-{max_thresh}",
                'samples_collected': len(samples),
                'target_samples': self.samples_per_threshold,
                'sample_ids': [s['id'] for s in samples] if samples else []
            }

        import json
        with open(f"{self.output_dir}/sampling_report.json", 'w') as f:
            json.dump(summary, f, indent=2)

    def run_analysis(self):
        """Run the complete threshold analysis."""
        logger.info("Starting comprehensive PDF threshold analysis...")

        # Process all WARC files
        self.process_all_warcs()

        # Save sample PDFs
        self.save_sample_pdfs()

        # Generate statistics
        df, stats = self.generate_statistics()

        # Create visualizations
        self.create_visualizations(df, stats)

        # Generate sampling report
        self.generate_sampling_report()

        logger.info("Analysis complete!")
        logger.info(f"Results saved to: {self.output_dir}/")
        logger.info(f"  - Sample PDFs: {self.output_dir}/samples/*/")
        logger.info(f"  - Statistics: {self.output_dir}/statistics.json")
        logger.info(f"  - Plots: {self.output_dir}/plots/")
        logger.info(f"  - Sampling report: {self.output_dir}/sampling_report.json")
        logger.info(f"\nNext steps:")
        logger.info(f"  1. Review the plots to understand OCR probability distribution")
        logger.info(f"  2. Manually evaluate sample PDFs in each threshold category")
        logger.info(f"  3. Determine appropriate threshold for OCR vs text extraction")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Comprehensive PDF threshold analysis')
    parser.add_argument('--data-folder', default='spec/phase3/data', help='Folder containing WARC files')
    parser.add_argument('--model-path', default='spec/phase3/data/pdf_classifier_real_data.xgb', help='Path to trained model')
    parser.add_argument('--samples-per-threshold', type=int, default=5, help='Number of samples per threshold')

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model_path):
        logger.info(f"ERROR: Model not found at {args.model_path}")
        logger.info("Please run train_pdf_classifier.py first to create the model")
        return

    # Check if data folder exists
    if not os.path.exists(args.data_folder):
        logger.info(f"ERROR: Data folder not found at {args.data_folder}")
        return

    analyzer = PDFThresholdAnalyzer(
        data_folder=args.data_folder,
        model_path=args.model_path
    )
    analyzer.samples_per_threshold = args.samples_per_threshold

    analyzer.run_analysis()


if __name__ == "__main__":
    main()
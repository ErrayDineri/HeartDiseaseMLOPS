import json
from pathlib import Path

import pandas as pd


def main():
    csv_path = Path('data.csv')
    if not csv_path.exists():
        raise FileNotFoundError('data.csv introuvable')

    df = pd.read_csv(csv_path)

    report = {
        'shape': {'rows': int(df.shape[0]), 'cols': int(df.shape[1])},
        'columns': df.columns.tolist(),
        'missing_values': df.isna().sum().to_dict(),
        'duplicate_rows': int(df.duplicated().sum()),
        'target_distribution': df['target'].value_counts().to_dict() if 'target' in df.columns else {},
        'numeric_summary': df.describe().to_dict()
    }

    output_json = Path('data_quality_report.json')
    output_json.write_text(json.dumps(report, indent=2), encoding='utf-8')

    md = []
    md.append('# Data Quality Report')
    md.append('')
    md.append(f"- Shape: {df.shape[0]} lignes x {df.shape[1]} colonnes")
    md.append(f"- Lignes dupliquées: {report['duplicate_rows']}")
    md.append('')
    md.append('## Valeurs manquantes')
    for k, v in report['missing_values'].items():
        md.append(f'- {k}: {v}')
    md.append('')
    if report['target_distribution']:
        md.append('## Distribution target')
        for k, v in report['target_distribution'].items():
            md.append(f'- Classe {k}: {v}')

    Path('data_quality_report.md').write_text('\n'.join(md), encoding='utf-8')
    print('data_quality_report.json et data_quality_report.md générés')


if __name__ == '__main__':
    main()

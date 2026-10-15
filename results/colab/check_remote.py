from pathlib import Path
p=Path('/content/posedflow/bundle.sha256')
print('BUNDLE_MATCH' if p.exists() and p.read_text().strip() == '1d837501640d036c19a8de1e124cd43e1ff493996f5124399859214cd720b8d5' else 'BUNDLE_MISSING')

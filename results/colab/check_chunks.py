from pathlib import Path
for p in Path('/content').glob('posedflow-1d837501640d036c19a8de1e124cd43e1ff493996f5124399859214cd720b8d5-part-*'):
    print('CHUNK_EXISTS '+p.name+' '+str(p.stat().st_size))

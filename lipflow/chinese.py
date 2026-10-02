"""Install the author's optional CMLR research weights; never bundle them in a wheel."""
from pathlib import Path
import hashlib
import os
import tempfile
import zipfile

SOURCE = 'https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages'
LICENSE_URL = SOURCE + '/blob/master/LICENSE'
ARTIFACTS = {
    'vsr': ('1tN79KWj3Za8oS-7WWrcPMBCIx0RARfVD',
            'd4dd9974be9f685e58e6a3649c350b15385f32d34050b3572e73726915ca9b7b'),
    'lm': ('1de4lqf_SkkjVn4t_NFs1dmRhAQpR1CfD',
           '7872b2ef0bd26cbf524579125793a5edd9c2a9e4d2577b56b0242c0db159baf1'),
}


def install(accept_research_license=False, destination=None):
    if not accept_research_license:
        raise ValueError(f'CMLR weights are research-only. Read {LICENSE_URL} and pass --accept-research-license.')
    try:
        import gdown
    except ImportError as e:
        raise RuntimeError('Install the download extra first: uv sync --extra chinese') from e
    target = Path(destination or Path(__file__).resolve().parent.parent / 'models' / 'zh')
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=target) as tmp:
        staged = Path(tmp)
        for component, (file_id, digest) in ARTIFACTS.items():
            archive = staged / (component + '.zip')
            gdown.download(id=file_id, output=str(archive), quiet=False)
            if not archive.exists() or hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
                raise ValueError(f'{component}: checksum mismatch; no weights installed')
            with zipfile.ZipFile(archive) as z:
                for filename in ('model.json', 'model.pth'):
                    matches = [n for n in z.namelist() if Path(n).name == filename]
                    if len(matches) != 1:
                        raise ValueError(f'{component}: unexpected archive contents')
                    # Only known members are written; no archive paths are extracted.
                    out = staged / component / filename
                    out.parent.mkdir(exist_ok=True)
                    out.write_bytes(z.read(matches[0]))
        for component in ARTIFACTS:
            (target / component).mkdir(exist_ok=True)
            for filename in ('model.json', 'model.pth'):
                os.replace(staged / component / filename, target / component / filename)
    return target

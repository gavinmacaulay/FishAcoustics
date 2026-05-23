"""Helps convert EK80 calibration files into the format that LSSS wants."""

from pathlib import Path
from xml.etree import ElementTree as ET

ek80_cal_dir = Path(r'C:\Users\GavinMacaulay\Data - not synced\temp\cal files')

cal_files = ek80_cal_dir.glob('*.xml')

def to_list(text):
    return [float(x) for x in text.split(';')]

for cal_file in cal_files:
    tree = ET.parse(cal_file)
    root = tree.getroot()
    freq = to_list(root.find('.//CalibrationResults/Frequency').text)
    gain = to_list(root.find('.//CalibrationResults/Gain').text)


    print(f'Writing to {cal_file.stem}')
    with open(cal_file.with_suffix('.lsss'), 'w') as ff:
        ff.write('<broadband>\n')
        ff.write('  <g>\n')
        for f, g in zip(freq, gain):
            ff.write(f'    <case hz="{int(f)}" g="{g:.2f}"\n')
        ff.write('  </g>\n')
        ff.write('</broadband>\n')

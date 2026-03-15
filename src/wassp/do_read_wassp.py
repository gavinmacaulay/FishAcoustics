"""Convert WASSP files into EK60-formatted files.
# Extracts the echosounder data and writes out as Simrad EK60 data files.
# If desired, multiple echosounder beams get saved as multiple
# channels in the EK60 files. Plenty of hard-coded parameters here to make things
# work enough for the purpose.
"""
# /// 
# dependencies = [
#   "numpy"
# ]
# ///

from pathlib import Path
import numpy as np
import struct
from dataclasses import dataclass
from datetime import datetime, timedelta

projectDir = Path(r'D:\Working\Projects\Active\2022 WindFarms')

dataDir = projectDir.joinpath('data').joinpath('2023_wassp_files')

# the .raw files can either go into the same directory as the wassp files if resultsDir is None, or into a common directory
resultsDir = projectDir.joinpath('data').joinpath('2023_ek60_files')

files = sorted(dataDir.glob('**/*.wmbf'))
default_gain = 13 # [dB]
default_pulse_duration = 0.001 # [s]
default_frequency = 160e3 # [Hz]

#%%
epoch = datetime(1601,1,1)
def datetimeToEK60Time(wasspTime):
    # convert Python datetime to 100 nanoseconds since January 1, 1601
    # as a 64 bit unsigned integer and express in Windows 64-bit time 
    ek60Time = int((wasspTime - epoch).total_seconds()*1e7)
    
    return ek60Time

def datetimeFromWasspTime(ping_date, timeofday):
    # generate a Python datetime from the given values. Timeofday
    # is in nanoseconds since midnight
    timestamp = ping_date + timedelta(microseconds=timeofday*1e-3)
    return timestamp

class EK60datagram(object):
    
    def __init__(self, dg_type, dg_timestamp: datetime):
        self.headerlength = 12
        self.lengthlength = 8 # 4 bytes at each end
        self.type = dg_type
        self.timestamp = dg_timestamp
        self.length = 0
        
    def write_header(self, fid):
        #print(f'Writing datagram of {self.length} bytes')
        fid.write(struct.pack('<l', self.length))
        fid.write(self.type.encode())
        fid.write(struct.pack('<Q', datetimeToEK60Time(self.timestamp)))
        
    def write_trailer(self, fid):
        fid.write(struct.pack('<l', self.length))
        
    def __str__(self):
        return f'Datagram of type {self.type} and length {self.length} bytes.'
   
class EK60nmea(EK60datagram):
    def __init__(self, dg_timestamp, nmea):
        EK60datagram.__init__(self, 'NME0', dg_timestamp)
        self.nmea = nmea.strip()
        self.length = self.headerlength + len(self.nmea)
        
    def write(self, fid):
        self.write_header(fid)        
        fid.write(self.nmea.encode('ASCII'))
        self.write_trailer(fid)

class EK60raw(EK60datagram):
    def __init__(self, dg_timestamp, raw):
        EK60datagram.__init__(self, 'RAW0', dg_timestamp)
        
        self.channel = 0
        self.mode = 1 # power data only
        self.transducerdepth = 0.0
        self.frequency = default_frequency # [Hz] placeholder
        self.transmitpower = 2000.0 # [W] placeholder
        self.pulselength = 0.001 # [s] placeholder
        self.bandwidth = 3000.0 # [Hz] placeholder
        self.sampleinterval = 0.0
        self.soundspeed = 0.0
        self.absorptioncoefficient = 0.0
        self.heave = 0.0
        self.txroll = 0.0
        self.txpitch = 0.0
        self.temperature = 0.0
        self.spare1 = 0
        self.spare2 = 0
        self.rxroll = 0.0
        self.rxpitch = 0.0
        self.offset = 0
        self.count = len(raw)
        self.power = np.round(raw / (10*np.log10(2.0)/256.0)).astype('short')
        
        self.length = self.headerlength + 72 + 2*len(self.power)
    
    def write(self, fid):
        a=fid.tell()
        self.write_header(fid)
        fid.write(struct.pack('<hhffffffffffffhhffll', self.channel, self.mode, self.transducerdepth,
                              self.frequency, self.transmitpower, self.pulselength,
                              self.bandwidth, self.sampleinterval, self.soundspeed,
                              self.absorptioncoefficient, self.heave, self.txroll,
                              self.txpitch, self.temperature, self.spare1, self.spare2, 
                              self.rxroll, self.rxpitch, self.offset, self.count))
        self.power.tofile(fid)
        self.write_trailer(fid)
        #print(f'{fid.tell()-a-self.lengthlength}, cf {self.length}')
        
@dataclass
class EK60configtransducer(object):
    channel: str = format('WASSP echosounder', '\x00<128') # placeholder
    beamtype: int = 0
    frequency: float = default_frequency # [Hz] placeholder
    gain: float = default_gain # [dB] 
    equivalentbeamangle: float = -20.0 # [dB] placeholder
    bwalongship: float = 0.0
    bwathwartship: float = 0.0
    asalongship: float = 0.0
    asathwartship: float = 0.0
    offsetalongship: float = 0.0
    offsetathwartship: float = 0.0
    pulselength = np.array([1,2,3,4,5], dtype='float32')*default_pulse_duration
    spare1: str = ' '*8 
    gaintable = np.full(5, default_gain, dtype='float32')
    spare2: str = ' '*8
    sacorrection = np.full(5, 0.0, dtype='float32')
    spare3: str = ' '*8
    gptsoftwareversion: str = ' '*16
    spare4: str = ' '*28
    length: int = 320 # bytes
    
    def setChannelName(self, name):
        self.channel = format(name[:128], '\x00<128')
        
    def write(self, fid):
        bytesWritten = fid.tell()
        fid.write(self.channel.encode())
        fid.write(struct.pack('<lfffffffff', self.beamtype, self.frequency, self.gain,
                               self.equivalentbeamangle, self.bwalongship, 
                               self.bwathwartship, self.asalongship, self.asathwartship,
                               self.offsetalongship, self.offsetathwartship))
        fid.write(struct.pack('<ffffff', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        self.pulselength.tofile(fid)
        fid.write(self.spare1.encode())
        self.gaintable.tofile(fid)
        fid.write(self.spare2.encode())
        self.sacorrection.tofile(fid)
        fid.write(self.spare3.encode())
        fid.write(self.gptsoftwareversion.encode())
        fid.write(self.spare4.encode())
        bytesWritten = fid.tell() - bytesWritten
        #print(f'configtransducer: {bytesWritten}, cf {self.length}')
        
@dataclass
class EK60configheader(object):
    surveyname: str = ' '*128
    transectname: str = ' '*128
    soundername: str = format('WASSP', '<128')
    version: str = ' '*30
    spare: str = ' '*98
    transducercount: int = 1
    length: int = 516 # bytes
    
    def write(self, fid):
        bytesWritten = fid.tell()
        fid.write(self.surveyname.encode())
        fid.write(self.transectname.encode())
        fid.write(self.soundername.encode())
        fid.write(self.version.encode())
        fid.write(self.spare.encode())
        fid.write(struct.pack('l', self.transducercount))
        bytesWritten = fid.tell() - bytesWritten
        #print(f'configheader: {bytesWritten}, cf {self.length}')

    
class EK60configuration(EK60datagram):
    def __init__(self, dg_timestamp):
        EK60datagram.__init__(self, 'CON0', dg_timestamp)
        self.initialised = False
        
    def initialise(self, numchannels):
        if not self.initialised:
            self.initialised = True
            self.configheader = EK60configheader()
            self.configheader.transducercount = numchannels
            self.configtransducer = [EK60configtransducer() for i in range(self.configheader.transducercount)]
            self.length = self.headerlength + self.configheader.length
            for i in range(self.configheader.transducercount):
                self.length += self.configtransducer[i].length
            for i, t in enumerate(self.configtransducer):
                t.setChannelName(f'WASSP echosounder beam {i+1}')
                
        
    def write(self, fid):
        bytesWritten = fid.tell()
        self.write_header(fid)
        self.configheader.write(fid)
        for i in range(self.configheader.transducercount):
            self.configtransducer[i].write(fid)
        self.write_trailer(fid)
        bytesWritten = fid.tell() - bytesWritten - self.lengthlength
        #print(f'configuration: {bytesWritten}, cf {self.length}')


        
#%%
prev_file_ping_time = datetime(1601,1,1)

for file in files:
    print(f'Processing {file.name}')

    packet_types = []

    echogram = []
    plot_done = False

    ek60_config = EK60configuration(datetime(1601,1,1))
    ek60_datagrams = [ek60_config]
    ii = 0
    haveValidDate = False
    ping_date = datetime(1601, 1, 1)
    prev_timeofday = -1
    
    sonar_freq = []
    sonar_bandwidth = []
    
    with open(file, mode='rb') as f:
        while True:
            packet_start = f.read(4)
            if not packet_start: # eof
                break
        
            magic_start, = struct.unpack('<L', packet_start)
            length, = struct.unpack('<L', f.read(4))
            packet_type = struct.unpack('8s', f.read(8))[0].decode('ascii')
            packet_version, = struct.unpack('<L', f.read(4))
            msg_flags, = struct.unpack('<L', f.read(4))
            packet = f.read(length - 28)
            magic_end, = struct.unpack('<L', f.read(4))
        
            packet_types.append(packet_type)
            #print(f'Read packet: {magic_start:X}, {length}, {packet_type}, {magic_end:X}')
            ii += 1
            # if ii > 50:
            #    break
                
            if packet_type == 'SONADISP':
                i = 92
                # (_, timestamp, pingno, lat, long, bearing, sample_rate, c, 
                #  alpha, spreading, N, M, tx_power, pulse_width, sample_type, 
                #  sample_offset, _, _, _) = struct.unpack('<QQIddfffffIIfIIIIII', packet[:i])
                # # sort out some units
                # pulse_width = pulse_width * 1e-9 # [s]
                
                # i += N*4 # skip over N reserved U32's
                
                # detection_point = np.frombuffer(packet[i:], dtype='uint32', count=N)
                # i += N*4
                
                # beam_angle = np.frombuffer(packet[i:], dtype='float32', count=N)
                # i += N*4
                
                # sample_interval = c / sample_rate / 2 # [m]
                
                # data = np.frombuffer(packet[i:], dtype='int16', count=N*M)
                # data = np.reshape(data, (N,M)).T / 128.0 # [dB]

                # if not plot_done:
                #     ax = plt.subplot(111, projection='polar')
                #     ax.set_theta_direction(-1) # clockwise
                #     ax.set_theta_offset(-np.pi/2)
                #     ax.set_frame_on(False)
                #     r = np.arange(0, data.shape[0])*sample_interval
                #     ax.pcolormesh(beam_angle*np.pi/180, r, data, shading='auto')
                #     ax.xaxis.set_ticklabels([])
                #     ax.grid(axis='y', linestyle=':', color='grey')
                #     plt.savefig(dataDir.joinpath(f'{file.stem}_sonar_{pingno}.png'), bbox_inches='tight', pad_inches=0.1)
                #     plt.close()
                #     plot_done = True
                
            if packet_type == 'SONASTAT':
                i = 48
                (_, system_temp, transducer_temp, ping_rate, freq, bandwidth, ping_state,
                 c, tide, link_speed, progress, source, status) = struct.unpack('<QfffffIffIBBH', packet[:i])
                sonar_freq.append(freq)
                sonar_bandwidth.append(bandwidth)
                
            if packet_type == 'SENUPDAT':
                i = 16
                (_, year, month, day, hour, minute, millisecond) = struct.unpack('<QHBBBBH', packet[:i])

                if not haveValidDate:
                    ping_date = datetime(year, month, day)
                    ek60_config.timestamp = datetimeFromWasspTime(ping_date, (hour*60*60 + minute*60)*1e9 + millisecond*1e6)
                    haveValidDate = True
                #print(f'SENUPDAT {ek60_config.timestamp:%Y-%m-%d %H:%M:%S.%f}')
                                
            if packet_type == 'FISHDATA':
                if haveValidDate:
                    i = 84
                    
                    (_, timeofday, pingno, lat, long, bearing, sample_rate, c, 
                     alpha, spreading, N, M, sample_type, _, _, _, _) = struct.unpack('<QQIddfffffIIIIIII', packet[:i])
    
                    if timeofday < prev_timeofday:
                        print('went past midnight')
                        ping_date += timedelta(days=1)

                    prev_timeofday = timeofday
    
                    timestamp = datetimeFromWasspTime(ping_date, timeofday)
                    #print(f'FISHDATA {timestamp:%Y-%m-%d %H:%M:%S.%f}')
                    if timestamp > prev_file_ping_time:
                        alpha *= 1e-3 # convert from dB/km to dB/m
                        
                        white_line = np.frombuffer(packet[i:], dtype='uint32', count=N)                
                        i += N*4
                        
                        beam_angle = np.frombuffer(packet[i:], dtype='float32', count=N)
                        i += N*4
                        
                        beam_width = np.frombuffer(packet[i:], dtype='float32', count=N)
                        i += N*4
        
                        data = np.frombuffer(packet[i:], dtype='int16', count=N*M)
                        data = np.reshape(data, (N,M)).T / 128.0 # [dB]
                        
                        # remove the TVG
                        range_m = c * np.arange(data.shape[0]) / (sample_rate * 2)  # [m]
                        range_m[0] += 1e-3 # avoid a range of 0 for tvg reasons
                        tvg = spreading*np.log10(range_m) + 2*alpha*range_m
                        data = (data.T - tvg.T).T
                        beamsToSave = range(N) # save all beams
                        
                        ek60_config.initialise(N) # only gets run the first time
        
                        for beam in beamsToSave:
                            dg = EK60raw(timestamp, data[:,beam])
                            dg.channel = beam+1
                            dg.soundspeed = c
                            dg.absorptioncoefficient = alpha
                            dg.sampleinterval = 1.0/sample_rate # convert from Hz to interval [s]
                            dg.frequency = ek60_config.configtransducer[beam].frequency
                        
                            ek60_config.configtransducer[beam].bwalongship = beam_width[beam]
                            ek60_config.configtransducer[beam].bwathwartship = beam_width[beam]
                        
                            ek60_datagrams.append(dg)
                            
                        # accumlate a beam for an echogram
                        echogram.append(data[:,2]) # beam 2 is the downwards one when there are five.
                        
                        # plt.imshow(data, aspect='auto')
                        # plt.xlabel('Beam number')
                        # plt.ylabel('Sample number')
                        # plt.savefig(dataDir.joinpath(f'{file.stem}_{pingno}.png'), bbox_inches='tight', pad_inches=0.1)
                        # plt.close()
            elif packet_type == 'RAW_SENS':
                
                if haveValidDate:
                    i = 24
                    (_, timeofday, port, protocol, _, N) = struct.unpack('<QQBBHI', packet[:i])
                                        
                    timestamp = datetimeFromWasspTime(ping_date, timeofday)
                    sens_data = np.frombuffer(packet[i:], dtype='uint8', count=N)
                    if sens_data[0] == ord('$'):
                        ek60_datagrams.append(EK60nmea(timestamp, sens_data.tobytes().decode()))
                    #print(f'RAW_SENS {timestamp:%Y-%m-%d %H:%M:%S.%f}')
                
    # make an echogram
        
    # ee = np.array(echogram).T
    # plt.figure()
    # im = plt.imshow(ee, aspect='auto', extent=(0, ee.shape[1], ee.shape[0]*sample_interval, 0))
    # cb = plt.colorbar(im)
    # cb.set_label('Power [dB]')
    # plt.xlabel('Ping number')
    # plt.ylabel('Range [m]')
    # plt.savefig(dataDir.joinpath(f'{file.stem}_echogram_{pingno}.png'), bbox_inches='tight', pad_inches=0.1)
    # plt.close()
    
    # Add some info to the datagrams that isn't available until later
    
    # changes from file to file, which is inconvenient in LSSS, so don't do that change
    mean_freq = np.array(sonar_freq).mean()
    mean_bandwidth = np.array(sonar_bandwidth).mean()
    for dg in ek60_datagrams:
        # if dg.type == 'CON0':
        #     for i in dg.configtransducer:
        #         i.frequency = mean_freq
        if dg.type == 'RAW0':
            #dg.frequency = mean_freq 
            dg.bandwidth = mean_bandwidth
            prev_file_ping_time = dg.timestamp
    
    # write out the ek60_datagrams into an EK60 data file.
    first_ping = None
    last_ping = None
    counter_GGA = 0
    counter_HDT = 0
    counter_VTG = 0
    every = 20 # only output every every'th NMEA message
    
    if resultsDir:
        rawfile = resultsDir.joinpath(file.name).with_suffix('.raw') # all into one directory
    else:
        rawfile = file.with_suffix('.raw') # same dir as wassp file
    
    with open(rawfile, 'wb') as f:
        for dg in ek60_datagrams:

            if dg.type == 'RAW0':
                if not first_ping:
                    first_ping = dg.timestamp
                last_ping = dg.timestamp
                dg.write(f)
            elif dg.type == 'NME0':
                if dg.nmea[3:6] == 'GGA':
                    counter_GGA += 1
                    if not (counter_GGA % every):
                        dg.write(f)
                        #print(f'{dg.timestamp:%Y-%m-%d %H:%M:%S.%f} {dg.nmea}')
                elif dg.nmea[3:6] == 'HDT':
                    counter_HDT += 1
                    if not (counter_HDT % every):
                        dg.write(f)
                elif dg.nmea[3:6] == 'VTG':
                    counter_VTG += 1
                    if not (counter_VTG % every):
                        dg.write(f)
                else:
                    dg.write(f)
            else:
                dg.write(f)
                
    print(f'Wrote {len(ek60_datagrams)} EK60 datagrams')
    
    try:
        file.with_suffix('.idx').unlink() # so that LSSS will make a new one
    except:
        pass
    
    print(f'First timestamp in file      {ek60_datagrams[0].timestamp:%Y-%m-%d %H:%M:%S.%f}')
    print(f'First ping timestamp in file {first_ping:%Y-%m-%d %H:%M:%S.%f}')
    print(f'Last ping timestamp in file  {last_ping:%Y-%m-%d %H:%M:%S.%f}')
    print(f'Last timestamp in file       {ek60_datagrams[-1].timestamp:%Y-%m-%d %H:%M:%S.%f}')
    

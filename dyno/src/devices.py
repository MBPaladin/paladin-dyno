import ctypes
import math
import numpy as np
import struct
import xmltodict
import yaml
import queue
import threading
from deployment import dyno_paths

# Hard coded device ID's to double check during bringup
BECKHOFF_VENDOR_ID = 2
KOLLMORGEN_VENDOR_ID = 106
EK1100_PRODUCT_CODE = 72100946
EL2002_PRODUCT_CODE = 131215442
EL2502_PRODUCT_CODE = 163983442     # Hex: 0x02503052
EL3208_PRODUCT_CODE = 210251858    # Hex: 0x0c883052
EL5042_PRODUCT_CODE = 330444882    # Hex: 0x14f23052
EL2004_PRODUCT_CODE = 131346514    # Hex: 0x07d43052 (Standard 24V variant)
EL2024_PRODUCT_CODE = 132657234    # Hex: 0x07e83052 (Standard 12V variant)
ELM3004_PRODUCT_CODE = 1344368073  # Hex: 0x50222e11
EL1002_PRODUCT_CODE = 65679442
ELM3002_PRODUCT_CODE = 1344368041
AKD_PRODUCT_CODE = 4279108
AXON_PRODUCT_CODE = 2454913024

# utilit classes

class Unwrapper:
    def __init__(self, fs_counts):
        self.fs_counts = fs_counts
        self.past_count = None
        self.wraps = 0

    def __call__(self, counts):
        if self.past_count == None:
            self.past_count = counts
            return counts
        
        else:
            if self.past_count < 0.1*self.fs_counts and counts > 0.9*self.fs_counts:
                self.wraps -= 1
            elif self.past_count > 0.9*self.fs_counts and counts < 0.1*self.fs_counts:
                self.wraps += 1

            self.past_count = counts
            return counts + self.wraps*self.fs_counts

# At a base level, each ethercat device, even the bus couplers, needs to have a class containing a vendor and product ID to ensure that devices layouts match between hw / sw
class EK1100:
    def __init__(self, slave, name, params=None):
        self.vendor_id = BECKHOFF_VENDOR_ID
        self.product_id = EK1100_PRODUCT_CODE

class EL2002:
    def __init__(self, slave, name, params=None):
        self.vendor_id = BECKHOFF_VENDOR_ID
        self.product_id = EL2002_PRODUCT_CODE

class EL2502:
    class RxPDO(ctypes.Structure):
        _pack_ = 1
        _fields_ = [
            ('ch1_pwm', ctypes.c_uint16), # 16-bit PWM width for Channel 1
            ('ch2_pwm', ctypes.c_uint16), # 16-bit PWM width for Channel 2
        ]

    def __init__(self, slave, name, params=None):
        self._slave = slave
        self.name = name
        self.params = params
        
        # Instantiate the structure and default the outputs to 0
        self._rx_pdo = self.RxPDO()
        self._rx_pdo.ch1_pwm = 0
        self._rx_pdo.ch2_pwm = 0

    def setup(self):
        # 1. Clear any weird default assignments
        self._slave.sdo_write(index=0x1C12, subindex=0, data=(0x00).to_bytes(1, 'little'))  
        self._slave.sdo_write(index=0x1C13, subindex=0, data=(0x00).to_bytes(1, 'little'))  # Clear TxPDOs (we don't need input diagnostics)

        # 2. Explicitly map the standard RxPDOs for CH1 (0x1600) and CH2 (0x1601)
        self._slave.sdo_write(index=0x1C12, subindex=1, data=(0x1600).to_bytes(2, 'little')) 
        self._slave.sdo_write(index=0x1C12, subindex=2, data=(0x1601).to_bytes(2, 'little')) 
        
        # 3. Set count to 2 mapped output PDOs
        self._slave.sdo_write(index=0x1C12, subindex=0, data=(0x02).to_bytes(1, 'little'))  

    def process_txpdo(self):
        # We cleared 0x1C13, so no inputs to process
        pass

    def write_rxpdo(self):
        # We MUST write our 4-byte buffer (even if it's just zeroes) to the master's payload
        self._slave.output = bytes(self._rx_pdo)

class EL1002:
    class TxPDO(ctypes.Structure):
        _pack_ = 1
        _fields_ = [
            ('inputs', ctypes.c_uint8), # 1 byte for 2 digital inputs (bits 0 and 1)
        ]

    def __init__(self, slave, name, params=None):
        self.vendor_id = BECKHOFF_VENDOR_ID
        self.product_id = EL1002_PRODUCT_CODE
        self._tx_pdo = self.TxPDO()
        self._slave = slave

    def process_txpdo(self):
        if len(self._slave.input) == ctypes.sizeof(self._tx_pdo):
            self._tx_pdo = self.TxPDO.from_buffer_copy(self._slave.input)
        self.state = [bool((self._tx_pdo.inputs >> channel) & 0x01) for channel in range(2)]

class EL2004:
    # Class-level counter to track EtherCAT's automatic bit-packing
    _output_bit_counter = 0

    def __init__(self, slave, name, params=None):
        self._slave = slave
        self.name = name
        self.params = params

        self.data_counter = 0
        
        # Internal state tracking for the 4 channels (False = OFF, True = ON)
        self.channels = [False, False, False, False] 

        # Calculate our bit shift (will alternate between 0 and 4)
        self.bit_shift = EL2004._output_bit_counter % 8
        
        # Increment the shared counter for the next terminal
        EL2004._output_bit_counter += 4

    def setup(self):
        # Default PDO mappings are natively sufficient.
        pass

    def set_channel(self, channel, state):
        """
        Set the state of a specific channel.
        :param channel: Integer from 1 to 4
        :param state: Boolean (True for ON, False for OFF)
        """
        if 1 <= channel <= 4:
            self.channels[channel - 1] = bool(state)
        else:
            print(f"WARNING: {self.name} invalid channel {channel}. Must be 1-4.")

    def write_rxpdo(self):
        '''Safely packs the boolean states into the shared master IO map without overwriting neighbors.'''

        # 1. Calculate our desired 4-bit state
        output_bits = 0
        for i in range(4):
            if self.channels[i]:
                output_bits |= (1 << i)
                
        # 2. Read the current shared byte from the master's IO map
        current_output = self._slave.output
        if not current_output:
            current_byte = 0
        else:
            current_byte = current_output[0]

        # 3. Create a mask to clear ONLY our 4 bits 
        # (e.g., if shift is 4, mask is 0x0F. If shift is 0, mask is 0xF0)
        clear_mask = ~(0x0F << self.bit_shift) & 0xFF

        # 4. Clear our bits in the shared byte, then insert our new state
        current_byte = (current_byte & clear_mask) | (output_bits << self.bit_shift)

        # 5. Write the safely modified byte back to the PySOEM buffer
        self._slave.output = bytes([current_byte])

class ELM3002:
    class TxPDO(ctypes.Structure):
        _pack_ = 1 # Ensures no padding between fields
        _fields_ = [
            ('ch1_sample_count', ctypes.c_uint8),
            ('ch1_status_word', ctypes.c_uint8),
            ('ch1_padding', ctypes.c_uint16),
            ('ch1_input_value', ctypes.c_int32),
            ('ch2_sample_count', ctypes.c_uint8),
            ('ch2_status_word', ctypes.c_uint8),
            ('ch2_padding', ctypes.c_uint16),
            ('ch2_input_value', ctypes.c_int32),
        ]

    def __init__(self, slave, name, params):
        self._slave = slave
        self._tx_pdo = self.TxPDO() # Slave -> Master (Input from terminal)
        self.name = None
        self.params = params
        self.input_torque = 0
        self.output_torque = 0

    def setup(self):
        #Set 8000 and 8010 to corresponding measurement ranges: 2 = +- 10v, 3 = +- 5v
        param_objs = [0x8000, 0x8010]
        for ui, key in enumerate(self.params.keys()):
            if self.params[key]['fs_v'] == 10:
                self._slave.sdo_write(index=param_objs[ui], subindex=1, data=0x0002.to_bytes(2, 'little'))
            elif self.params[key]['fs_v'] == 5:
                self._slave.sdo_write(index=param_objs[ui], subindex=1, data=0x0003.to_bytes(2, 'little'))

        self._slave.sdo_write(index=0x1C12, subindex=0, data=(0x0000).to_bytes(2, 'little'))  # clear current TxPDO mapping
        self._slave.sdo_write(index=0x1C13, subindex=0, data=(0x0000).to_bytes(2, 'little'))  # clear current TxPDO mapping

        # Map PDO's
        pdo_obj = [0x1A00, 0x1A01, 0x1A21, 0x1A22]
        for ui, obj in enumerate(pdo_obj):
            self._slave.sdo_write(index=0x1C13, subindex=ui+1, data=(obj).to_bytes(2, 'little'))  
        self._slave.sdo_write(index=0x1C13, subindex=0, data=(0x0004).to_bytes(2, 'little'))

    def decode_status(self, status_word):
        sw_bits = bin(status_word)[2:]
        if len(sw_bits) < 8:
            sw_bits = '0'*(8 - len(sw_bits))+sw_bits

        status_codes = []

        if sw_bits[-1] == '1':
            status_codes.append('General Error')
        if sw_bits[-2] == '1':
            status_codes.append('Underrange')
        if sw_bits[-3] == '1':
            status_codes.append('Overrange')
        if sw_bits[-5] == '1':
            status_codes.append('Diagnostic message availible')
        if sw_bits[-6] == '1':
            status_codes.append('TxPDO Invalid state')

        return status_codes

    def process_txpdo(self):
        '''Reads the latest input bytes from the slave and populates the TxPDO structure.'''
        if len(self._slave.input) == ctypes.sizeof(self._tx_pdo):
            self._tx_pdo = self.TxPDO.from_buffer_copy(self._slave.input)
        else:
            print(f'WARNING: ELM3002 input buffer size mismatch! Expected {ctypes.sizeof(self._tx_pdo)}, Got {len(self._slave.input)}')
        
        raw_values = [
            self._tx_pdo.ch1_input_value,
            self._tx_pdo.ch2_input_value
        ]


        for i, ch_key in enumerate(['ch1', 'ch2']):
            ch_params = self.params.get(ch_key, {})
            ch_name = ch_params.get('name')
            
            if ch_name:
                fs_pos = ch_params.get('fs_pos', 1.0)
                offset = ch_params.get('offset', 0.0)
                
                # Formula for ELM 3000 series scaling
                scaled_value = (fs_pos * raw_values[i] / 7812500) + offset
                
                # This creates or updates the attribute (e.g., self.load_torque)
                setattr(self, ch_name, scaled_value)

class EL3208:
    class TxPDO(ctypes.Structure):
        _pack_ = 1 # Ensures no padding between fields
        _fields_ = [
            ('ch1_status_word', ctypes.c_uint16),
            ('ch1_value', ctypes.c_int16), 
            ('ch2_status_word', ctypes.c_uint16),
            ('ch2_value', ctypes.c_int16),
            ('ch3_status_word', ctypes.c_uint16),
            ('ch3_value', ctypes.c_int16), 
            ('ch4_status_word', ctypes.c_uint16),
            ('ch4_value', ctypes.c_int16),
            ('ch5_status_word', ctypes.c_uint16),
            ('ch5_value', ctypes.c_int16), 
            ('ch6_status_word', ctypes.c_uint16),
            ('ch6_value', ctypes.c_int16),
            ('ch7_status_word', ctypes.c_uint16),
            ('ch7_value', ctypes.c_int16), 
            ('ch8_status_word', ctypes.c_uint16),
            ('ch8_value', ctypes.c_int16),
        ]

    def __init__(self, slave, name, params):
        self._slave = slave
        self._tx_pdo = self.TxPDO() # Slave -> Master (Input from terminal)
        self.name = name
        self.params = params
        
        # Initialize all 8 channels
        self.temps_c = [0.0] * 8

    def setup(self):
        # Clear current TxPDO assignments in the Sync Manager
        self._slave.sdo_write(index=0x1C12, subindex=0, data=(0x00).to_bytes(1, 'little')) 
        self._slave.sdo_write(index=0x1C13, subindex=0, data=(0x00).to_bytes(1, 'little'))  

        param_objs = [0x8000, 0x8010, 0x8020, 0x8030, 0x8040, 0x8050, 0x8060, 0x8070]
        
        # Beckhoff EL3208 Sensor Element Type dictionary (Index 0x80n0:19)
        sensor_dict = {
            'PT100': 1,
            'NI100': 2,
            'PT1000': 3,
            'PT500': 4,
            'NI1000': 5,
            'NI120': 6,
            'RESISTANCE': 7 
        }
        
        for ui, key in enumerate(['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']):
            ch_params = self.params.get(key, {})
            raw_type = ch_params.get('sensor_type', 1) 
            
            # If the YAML provided a string, look it up. Otherwise, use the int/default.
            if isinstance(raw_type, str):
                sensor_type_int = sensor_dict.get(raw_type.upper(), 1) # Default to 1 (PT100) if typo
            else:
                sensor_type_int = int(raw_type)

            self._slave.sdo_write(index=param_objs[ui], subindex=0x19, data=sensor_type_int.to_bytes(2, 'little'))

        # Map the standard PDOs for CH1 through CH8
        pdo_obj = [0x1A00, 0x1A01, 0x1A02, 0x1A03, 0x1A04, 0x1A05, 0x1A06, 0x1A07]
        for ui, obj in enumerate(pdo_obj):
            self._slave.sdo_write(index=0x1C13, subindex=ui+1, data=(obj).to_bytes(2, 'little'))  
        
        # Set Sync Manager count to 8 mapped PDOs
        self._slave.sdo_write(index=0x1C13, subindex=0, data=(0x08).to_bytes(1, 'little'))

    def decode_status(self, status_word):
        sw_bits = bin(status_word)[2:]
        if len(sw_bits) < 16:
            sw_bits = '0'*(16 - len(sw_bits))+sw_bits

        status_codes = []

        if sw_bits[-1] == '1':
            status_codes.append('Underrange')
        if sw_bits[-2] == '1':
            status_codes.append('Overrange')
        if sw_bits[-7] == '1':
            status_codes.append('Error')
        if sw_bits[-15] == '1':
            status_codes.append('TxPDO Invalid Data')

        return status_codes

    def is_sensor_valid(self, status_word):
        """Helper to quickly check if a sensor is plugged in and reading valid data."""
        # Check bit 6 (Error) and bit 1 (Overrange) which usually trip on an open circuit.
        # Mask: 0x0040 (Bit 6) | 0x0002 (Bit 1)
        if status_word & 0x0042:
            return False
        return True

    def process_txpdo(self):
        '''Reads the latest input bytes from the slave and populates the TxPDO structure.'''
        if len(self._slave.input) == ctypes.sizeof(self._tx_pdo):
            self._tx_pdo = self.TxPDO.from_buffer_copy(self._slave.input)
        else:
            print(f'WARNING: EL3208 input buffer size mismatch! Expected {ctypes.sizeof(self._tx_pdo)}, Got {len(self._slave.input)}')

        # Group raw inputs and statuses
        channels = [
            (self._tx_pdo.ch1_status_word, self._tx_pdo.ch1_value),
            (self._tx_pdo.ch2_status_word, self._tx_pdo.ch2_value),
            (self._tx_pdo.ch3_status_word, self._tx_pdo.ch3_value),
            (self._tx_pdo.ch4_status_word, self._tx_pdo.ch4_value),
            (self._tx_pdo.ch5_status_word, self._tx_pdo.ch5_value),
            (self._tx_pdo.ch6_status_word, self._tx_pdo.ch6_value),
            (self._tx_pdo.ch7_status_word, self._tx_pdo.ch7_value),
            (self._tx_pdo.ch8_status_word, self._tx_pdo.ch8_value)
        ]
        
        ch_keys = ['ch1', 'ch2', 'ch3', 'ch4', 'ch5', 'ch6', 'ch7', 'ch8']

        for i, (status, value) in enumerate(channels):
            # 1. Calculate the real temperature
            if self.is_sensor_valid(status):
                temp = value * 0.1
            else:
                temp = math.nan
                
            self.temps_c[i] = temp # Keep updating the old generic list just in case
            
            # 2. NEW: Dynamically update the mapped sensor name (if one exists)
            ch_params = self.params.get(ch_keys[i], {})
            ch_name = ch_params.get('name')
            
            if ch_name:
                # E.g., setattr(self, 'load_stator_temp', 24.5)
                setattr(self, ch_name, temp)

class ELM3004:
    class TxPDO(ctypes.Structure):
        _pack_ = 1 # Ensures no padding between fields
        _fields_ = [
            # CH1
            ('ch1_sample_count', ctypes.c_uint8),
            ('ch1_status_word', ctypes.c_uint8),
            ('ch1_padding', ctypes.c_uint16),
            ('ch1_input_value', ctypes.c_int32),
            # CH2
            ('ch2_sample_count', ctypes.c_uint8),
            ('ch2_status_word', ctypes.c_uint8),
            ('ch2_padding', ctypes.c_uint16),
            ('ch2_input_value', ctypes.c_int32),
            # CH3
            ('ch3_sample_count', ctypes.c_uint8),
            ('ch3_status_word', ctypes.c_uint8),
            ('ch3_padding', ctypes.c_uint16),
            ('ch3_input_value', ctypes.c_int32),
            # CH4
            ('ch4_sample_count', ctypes.c_uint8),
            ('ch4_status_word', ctypes.c_uint8),
            ('ch4_padding', ctypes.c_uint16),
            ('ch4_input_value', ctypes.c_int32),
        ]

    def __init__(self, slave, name, params):
        self._slave = slave
        self._tx_pdo = self.TxPDO()
        self.name = name
        # If no params passed, ensure it's a dict so routing doesn't crash
        self.params = params if params is not None else {}

    def setup(self):
        # Clear current TxPDO mapping
        self._slave.sdo_write(index=0x1C12, subindex=0, data=(0x0000).to_bytes(2, 'little'))  
        self._slave.sdo_write(index=0x1C13, subindex=0, data=(0x0000).to_bytes(2, 'little'))  

        self._slave.sdo_write(index=0x1C33, subindex=1, data=(0x01).to_bytes(2, 'little'))

        # Set 8000, 8010, 8020, 8030 to corresponding measurement ranges: 2 = +- 10v, 3 = +- 5v
        param_objs = [0x8000, 0x8010, 0x8020, 0x8030]
        
        for ui, ch_key in enumerate(['ch1', 'ch2', 'ch3', 'ch4']):
            ch_params = self.params.get(ch_key, {})
            if ch_params.get('name', 'NA') != 'NA':
                fs_v = ch_params.get('fs_v', 10)
                if fs_v == 10:
                    self._slave.sdo_write(index=param_objs[ui], subindex=1, data=0x0002.to_bytes(2, 'little'))
                elif fs_v == 5:
                    self._slave.sdo_write(index=param_objs[ui], subindex=1, data=0x0003.to_bytes(2, 'little'))
                else:
                    print('Unsupported voltage range detected')

        # Map standard PDOs for channels 1 through 4
        pdo_obj = [0x1A00, 0x1A01, 0x1A21, 0x1A22, 0x1A42, 0x1A43, 0x1A63, 0x1A64]
        for ui, obj in enumerate(pdo_obj):
            self._slave.sdo_write(index=0x1C13, subindex=ui+1, data=(obj).to_bytes(2, 'little'))  
            
        # Set Sync Manager count to 4 mapped PDOs
        self._slave.sdo_write(index=0x1C13, subindex=0, data=(0x0008).to_bytes(2, 'little'))

    def decode_status(self, status_word):
        sw_bits = bin(status_word)[2:]
        if len(sw_bits) < 8:
            sw_bits = '0'*(8 - len(sw_bits))+sw_bits

        status_codes = []

        if sw_bits[-1] == '1':
            status_codes.append('General Error')
        if sw_bits[-2] == '1':
            status_codes.append('Underrange')
        if sw_bits[-3] == '1':
            status_codes.append('Overrange')
        if sw_bits[-5] == '1':
            status_codes.append('Diagnostic message available')
        if sw_bits[-6] == '1':
            status_codes.append('TxPDO Invalid state')

        return status_codes

    def process_txpdo(self):
        '''Reads the latest input bytes from the slave and populates the TxPDO structure.'''
        if len(self._slave.input) == ctypes.sizeof(self._tx_pdo):
            self._tx_pdo = self.TxPDO.from_buffer_copy(self._slave.input)
        else:
            print(f'WARNING: {self.name} input buffer mismatch!')

        raw_values = [
            self._tx_pdo.ch1_input_value,
            self._tx_pdo.ch2_input_value,
            self._tx_pdo.ch3_input_value,
            self._tx_pdo.ch4_input_value
        ]

        for i, ch_key in enumerate(['ch1', 'ch2', 'ch3', 'ch4']):
            ch_params = self.params.get(ch_key, {})
            ch_name = ch_params.get('name')
            
            if ch_name:
                fs_pos = ch_params.get('fs_pos', 1.0)
                offset = ch_params.get('offset', 0.0)
                
                # Formula for ELM 3000 series scaling
                scaled_value = (fs_pos * raw_values[i] / 7812500) + offset
                
                # This creates or updates the attribute (e.g., self.load_torque)
                setattr(self, ch_name, scaled_value)

class EL5042:
    class TxPDO(ctypes.Structure):
        _pack_ = 1 # Ensures no padding between fields
        _fields_ = [
            ('ch1_status_word', ctypes.c_uint16),
            ('ch1_position', ctypes.c_int64),
            ('ch2_status_word', ctypes.c_uint16),
            ('ch2_position', ctypes.c_int64),
        ]

    def __init__(self, slave, name, params = None):
        self._slave = slave
        self._tx_pdo = self.TxPDO() # Slave -> Master (Input from terminal)
        self.name = None
        self.positions = [0, 0]
        self.position = 0
        self.home = 0
        self._ch0_unwrapper = Unwrapper(2**32)
        self._ch1_unwrapper = Unwrapper(2**32)

    def setup(self):
        # set operating frequency to 10 MHz
        self._slave.sdo_write(index=0x8008, subindex=0x13, data=0x00.to_bytes(1, 'little')) #set count of mapped PDO's
        self._slave.sdo_write(index=0x8018, subindex=0x13, data=0x00.to_bytes(1, 'little')) #set count of mapped PDO's

        # write number of multiturn bits
        self._slave.sdo_write(index=0x8008, subindex=0x15, data=0x00.to_bytes(1, 'little')) #set count of mapped PDO's
        self._slave.sdo_write(index=0x8018, subindex=0x15, data=0x00.to_bytes(1, 'little')) #set count of mapped PDO's

        # write number of single turn bits
        self._slave.sdo_write(index=0x8008, subindex=0x16, data=0x20.to_bytes(1, 'little')) #set count of mapped PDO's
        self._slave.sdo_write(index=0x8018, subindex=0x16, data=0x20.to_bytes(1, 'little')) #set count of mapped PDO's

        self._slave.sdo_write(index=0x8008, subindex=3, data=0x01.to_bytes(1, 'little')) #set count of mapped PDO's
        self._slave.sdo_write(index=0x8018, subindex=3, data=0x01.to_bytes(1, 'little')) #set count of mapped PDO's

    def zero(self, actuator_position):
        self.home = self.position - actuator_position

    def process_txpdo(self):
        '''Reads the latest input bytes from the slave and populates the TxPDO structure.'''
        if len(self._slave.input) == ctypes.sizeof(self._tx_pdo):
            self._tx_pdo = self.TxPDO.from_buffer_copy(self._slave.input)
        else:
            print(f'WARNING: EL5042 input buffer size mismatch! Expected {ctypes.sizeof(self._tx_pdo)}, Got {len(self._slave.input)}')
        self.positions = [self._ch0_unwrapper(self._tx_pdo.ch1_position), self._ch1_unwrapper(self._tx_pdo.ch2_position)]
        self.position = -1 * self.positions[0] * (2*math.pi) / 2**32

class AKD:
    class RxPDO(ctypes.Structure):
        _pack_ = 1
        _fields_ = [
            ('controlword', ctypes.c_uint16),     # Offset 0  <-- STATE ENGINE AT FRONT
            ('torque_ff', ctypes.c_int16),        # Offset 2
            ('position_command', ctypes.c_int32), # Offset 4  (Aligned)
            ('velocity_command', ctypes.c_int32), # Offset 8  (Aligned)
            ('current_command', ctypes.c_int32),  # Offset 12 (Aligned)
            ('digital_outputs', ctypes.c_uint32), # Offset 16 (Aligned)
            ('control_mode', ctypes.c_int8),      # Offset 20
            ('padding', ctypes.c_int8),           # Offset 21
        ]

    class TxPDO(ctypes.Structure):
        _pack_ = 1
        _fields_ = [
            ('statusword', ctypes.c_uint16),       # 0x6041:00 (16 bits)
            ('op_mode', ctypes.c_int8), # 6061:00 (8 bit)
            ('padding', ctypes.c_int8),
            ('actual_position', ctypes.c_int32),    # 0x6064 (32 bits) Momentary actual value in increments, per FB1.Pscale
            ('actual_velocity', ctypes.c_int32),    # 0x606C:00 (32 bits) Velocity in milli-RPM
            ('actual_current', ctypes.c_int32),       # 0x2077:00 (32 bits) Measured current in mA
            ('i2t_counter', ctypes.c_uint32),       # 0x3427:03 (32 bits) I2T foldback
        ]

    def __init__(self, slave, name, params):
        self._slave = slave
        self.name = name
        self.params = params

        print('\n#### Configuring ',self.name,' ####\n')

        with open(dyno_paths.dyno_config_directory+'/absorbers.yaml', 'r') as f:
            absorber_params = yaml.safe_load(f)

        drive_name = self._slave.sdo_read(0x2031, 0).decode('utf-8')
        drive_name = drive_name.strip(drive_name[-1])

        if drive_name in [k for k in absorber_params.keys()]:
            self.params = absorber_params[drive_name]
            print('Loading Params for: ',drive_name)
        else:
            print(self.name,' drive params not found in absorber config. Reverting to dyno config params')

        self._rx_pdo = self.RxPDO()  # Master -> Slave (Output to drive)
        self._tx_pdo = self.TxPDO()  # Slave -> Master (Input from drive)

        self.pos_offset = 0
        self.velocity = 0
        self.position = 0
        self.current = 0

        self.torque_command = math.nan
        self.velocity_command = math.nan
        self.position_command = math.nan

        # Fault reset
        self._rx_pdo.controlword = 0x00FF # fault reset

        self.state = None
        self.sw_enable = False

        self.position_unwrapper = Unwrapper(fs_counts=2**32)

        self._rx_pdo.control_mode = 4
        self.mode = 'torque'
        self.switching_modes = False

        self.position_offset = 0
        self.flip_torque_sign = params['flip_torque_sign']
        self.flip_direction_sign = params.get('flip_direction_sign', False)

        self.torque_limit = self.params['motor_limits']['torque']
        self.velocity_limit = self.params['motor_limits']['velocity']
        self.acceleration_limit = self.params['motor_limits']['acceleration']
        self.rotatum_limit = self.params['motor_limits']['rotatum']

        self.fault = True
        self.enabled = False

        self.digital_out_state = 0x00000000
        self.mode_dict = {7:2,3:1,4:0,0:-1} #Converts from KM mode nomenclature to 0 = torque, 1 = velocity, 2 = position

    #This method cycles the servo drive through the operating and control modes
    def update_modes(self):
        sw = self._tx_pdo.statusword

        # Bit masks (DS402 status word)
        self.STATE_BITS = {
            'ready_to_switch_on': (sw & 0x0001) != 0,
            'switched_on':        (sw & 0x0002) != 0,
            'operation_enabled':  (sw & 0x0004) != 0,
            'fault':              (sw & 0x0008) != 0,
            'voltage_enabled':    (sw & 0x0010) != 0,
            'quick_stop':         (sw & 0x0020) != 0,
            'switch_on_disabled': (sw & 0x0040) != 0,
        }
                

        # Determine current DS402 state
        if self.STATE_BITS['fault']:
            if not self.state == 'fault':
                print(self.name, ' State: Fault')
            self.state = 'fault'

        elif self.STATE_BITS['switch_on_disabled']:
            if not self.state == 'sw_on_disable':
                print(self.name, ' State: Switch on Disabled')
            self.state = 'sw_on_disable'

        elif self.STATE_BITS['ready_to_switch_on'] and not self.STATE_BITS['switched_on']:
            if not self.state == 'ready_to_sw_on':
                print(self.name, ' State: Ready to Switch On')
            self.state = 'ready_to_sw_on'

        elif self.STATE_BITS['switched_on'] and not self.STATE_BITS['operation_enabled']:
            if not self.state == 'sw_on':
                print(self.name, ' State: Switched On')
            self.state = 'sw_on'
            
        elif self.STATE_BITS['operation_enabled']:
            if not self.state == 'enabled':
                print(self.name, ' State: Enabled')
            self.state = 'enabled'
        else:
            self.state = 'undefined'

        if self.switching_modes:
            print(self.state, self.target_mode, self.mode)
            if self.state == 'ready_to_sw_on': # only write _rx_pdo.control_mode when the drive is in the specific state. adjust the internal tracking of the operating mode
                if self.target_mode == 'position':
                    self._rx_pdo.control_mode = 7
                    self.pos_cmd_offset = self._tx_pdo.actual_position
                    self.mode = 'position'
                elif self.target_mode == 'velocity':
                    self._rx_pdo.control_mode = 3
                    self.mode = 'velocity'
                elif self.target_mode == 'torque':
                    self._rx_pdo.control_mode = 4
                    self.mode = 'torque'

            # Once drive confirms mode change, return to normal operation        
            if self.servo_drive_mode == 2 and self.target_mode == 'position':
                self.switching_modes = False      
            elif self.servo_drive_mode == 1 and self.target_mode == 'velocity':
                self.switching_modes = False
            elif self.servo_drive_mode == 0 and self.target_mode == 'torque':
                self.switching_modes = False

            # Confirm mode switch
            if not self.switching_modes:
                print(self.name,' Mode: ',self.target_mode)
                self.torque_command = math.nan
                self.velocity_command = math.nan
                self.position_command = math.nan

        # Always try to maintain atleast "ready to switch on" state
        if self.state in ['not_ready', 'sw_on_disable', 'undefined']:
            self._rx_pdo.controlword = 0x0006

        # State machine transitions to de-enable the drive when switching modes
        if self.switching_modes or not self.sw_enable:
            if self.state == 'enabled':
                self._rx_pdo.controlword = 0x0007
            elif self.state == 'sw_on':
                self._rx_pdo.controlword = 0x0006
            elif self.state == 'ready_to_sw_on':
                self._rx_pdo.controlword = 0x0006 # Maintain ready to switch on state

        # State machine transitions to enable the servo drive
        elif self.sw_enable:
            if self.state == 'ready_to_sw_on':
                self._rx_pdo.controlword = 0x0007 # Command transition 3
            elif self.state == 'sw_on':
                self._rx_pdo.controlword = 0x001F # Command transition 4
            elif self.state == 'enabled':
                self._rx_pdo.controlword = 0x001F # Maintain enable command

        # Shutdown if unreccognizable state recieved
        if self.state == 'undefined':
            self._rx_pdo.controlword = 0x0006

        self.fault = self.STATE_BITS['fault']
        self.enabled = self.state == 'enabled'

    # This method intakes a new target control mode
    def command_operating_mode(self, mode):
        print(self.name,' Switching from  ',self.mode,' to ',mode,' mode')
        # Snap position command to actual before de-enabling to zero following error
        if self.mode == 'position':
            self._rx_pdo.position_command = self._tx_pdo.actual_position
        self.target_mode = mode
        self.switching_modes = True # by setting this flag to true, the above switching logic will commence

        self.torque_command = math.nan
        self.velocity_command = math.nan
        self.position_command = math.nan

    def setup(self):

        sdo_data = self._slave.sdo_read(0x3598, 0, size=4) # Ensure size is 4 for int32
        print('IL.KP', int.from_bytes(sdo_data, byteorder='little', signed=False))

        # Kollmorgen has some interesting restrictions on PDO mapping. read their ethercat communications manual before adjusting things here.

        # Hex 0x00030000 = (1 << 16) | (1 << 17)
        mask_val = 0x00030000
        self._slave.sdo_write(index=0x60FE, subindex=2, data=mask_val.to_bytes(4, 'little'))


        # Clear PDO assignments
        for pdo_obj in [0x1C12, 0x1C13, 0x1600, 0x1601, 0x1602, 0x1603, 0x1A00, 0x1A01, 0x1A02, 0x1A03]:
            self._slave.sdo_write(index=pdo_obj, subindex=0, data=0x00.to_bytes(1, 'little'))

        # Build Rx PDOs
        self._slave.sdo_write(index=0x1600, subindex=1, data=0x60400010.to_bytes(4, 'little')) # Controlword (16-bit)
        self._slave.sdo_write(index=0x1600, subindex=2, data=0x60B20010.to_bytes(4, 'little')) # Torque FF (16-bit)
        self._slave.sdo_write(index=0x1600, subindex=3, data=0x60C10120.to_bytes(4, 'little')) # Pos Cmd (32-bit)
        self._slave.sdo_write(index=0x1600, subindex=0, data=(3).to_bytes(1, 'little'))

        self._slave.sdo_write(index=0x1601, subindex=1, data=0x60FF0020.to_bytes(4, 'little'))
        self._slave.sdo_write(index=0x1601, subindex=2, data=0x20710020.to_bytes(4, 'little'))
        self._slave.sdo_write(index=0x1601, subindex=0, data=(2).to_bytes(1, 'little'))

        self._slave.sdo_write(index=0x1602, subindex=1, data=0x60FE0120.to_bytes(4, 'little')) # DigOut (32-bit)
        self._slave.sdo_write(index=0x1602, subindex=2, data=0x60600008.to_bytes(4, 'little')) # Mode (8-bit)
        self._slave.sdo_write(index=0x1602, subindex=3, data=0x00020008.to_bytes(4, 'little')) # 8-bit Pad
        self._slave.sdo_write(index=0x1602, subindex=0, data=(3).to_bytes(1, 'little'))
        
        # Build Tx PDOs
        self._slave.sdo_write(index=0x1A00, subindex=1, data=0x60410010.to_bytes(4, 'little'))
        self._slave.sdo_write(index=0x1A00, subindex=2, data=0x60610008.to_bytes(4, 'little'))
        self._slave.sdo_write(index=0x1A00, subindex=3, data=0x20020108.to_bytes(4, 'little'))
        self._slave.sdo_write(index=0x1A00, subindex=4, data=0x60630020.to_bytes(4, 'little'))
        self._slave.sdo_write(index=0x1A00, subindex=0, data=0x04.to_bytes(1, 'little')) #set count of mapped PDO's

        self._slave.sdo_write(index=0x1A01, subindex=1, data=0x606C0020.to_bytes(4, 'little'))
        self._slave.sdo_write(index=0x1A01, subindex=2, data=0x20770020.to_bytes(4, 'little'))
        self._slave.sdo_write(index=0x1A01, subindex=0, data=0x02.to_bytes(1, 'little')) #set count of mapped PDO's
        
        self._slave.sdo_write(index=0x1A02, subindex=1, data=0x34270320.to_bytes(4, 'little'))
        self._slave.sdo_write(index=0x1A02, subindex=0, data=0x01.to_bytes(1, 'little')) #set count of mapped PDO's

        # Map Rx / Tx PDOs
        self._slave.sdo_write(index=0x1C12, subindex=1, data=0x1600.to_bytes(2, 'little')) # Link RxPDO 0x1600
        self._slave.sdo_write(index=0x1C12, subindex=2, data=0x1601.to_bytes(2, 'little')) # Link RxPDO 0x1600
        self._slave.sdo_write(index=0x1C12, subindex=3, data=0x1602.to_bytes(2, 'little')) # Link RxPDO 0x1600

        self._slave.sdo_write(index=0x1C13, subindex=1, data=0x1A00.to_bytes(2, 'little')) # Link TxPDO 0x1A00
        self._slave.sdo_write(index=0x1C13, subindex=2, data=0x1A01.to_bytes(2, 'little')) # Link TxPDO 0x1A01
        self._slave.sdo_write(index=0x1C13, subindex=3, data=0x1A02.to_bytes(2, 'little')) # Link TxPDO 0x1A02

        self._slave.sdo_write(index=0x1C12, subindex=0, data=0x03.to_bytes(1, 'little')) # Set count to 1 (one PDO linked)
        self._slave.sdo_write(index=0x1C13, subindex=0, data=0x03.to_bytes(1, 'little')) # Set PDO count to 2

        # Set units for servo drive
        self._slave.sdo_write(index=0x3660, subindex=0, data=0x0000.to_bytes(4, 'little'))
        self._slave.sdo_write(index=0x3659, subindex=0, data=0x0000.to_bytes(4, 'little'))

        # Set interpolation time
        val, power = 1, -3
        self._slave.sdo_write(index=0x60C2, subindex=1, data=val.to_bytes(1, 'little', signed=False))
        self._slave.sdo_write(index=0x60C2, subindex=2, data=power.to_bytes(1, 'little', signed=True))

    def set_dout(self, pin, state):
        """
        Sets AKD Digital Output.
        :param pin: 1 or 2
        :param state: True (On) or False (Off)
        """
        bit = 15 + pin # Pin 1 = Bit 16, Pin 2 = Bit 17
        if state:
            self.digital_out_state |= (1 << bit)
        else:
            self.digital_out_state &= ~(1 << bit)

    def process_txpdo(self):
        '''Reads the latest input bytes from the slave and populates the TxPDO structure.'''
        if len(self._slave.input) == ctypes.sizeof(self._tx_pdo):
            self._tx_pdo = self.TxPDO.from_buffer_copy(self._slave.input)
        else:
            print(f'WARNING: AKD input buffer size mismatch! Expected {ctypes.sizeof(self._tx_pdo)}, Got {len(self._slave.input)}')

        # zero out the drives position
        if not self._tx_pdo.actual_position == 0 and self.pos_offset == 0:
            self.pos_offset = self._tx_pdo.actual_position

        # convert +- pi position to [0, 2*pi)
        self.unwrapped_ticks = self.position_unwrapper(self._tx_pdo.actual_position+2**31)-2**31
        if self.flip_direction_sign:
            self.unwrapped_ticks = -self.unwrapped_ticks
        self.position = 2*math.pi*(self.unwrapped_ticks)/2**32 + self.position_offset # units = rev

        self.velocity = self._tx_pdo.actual_velocity / 1000 * (2*math.pi)/60# units = Rad/s
        if self.flip_direction_sign:
            self.velocity = -self.velocity
        self.current = self._tx_pdo.actual_current / 1000 # units = A_rms
        self.servo_drive_mode = self.mode_dict[self._tx_pdo.op_mode]

        # Update the command word
        self.update_modes()

        self._rx_pdo.digital_outputs = self.digital_out_state

        # Zero out RX entries, they'll be filled in later
        self._rx_pdo.current_command = 0
        self._rx_pdo.position_command = 0
        self._rx_pdo.velocity_command = 0
        self._rx_pdo.torque_ff = 0


    def write_rxpdo(self):
        self._slave.output = bytes(self._rx_pdo)

    def send_command(self, command, torque_ff = 0):

        # position commands are scaled over a base of 2^32 counts / rev
        if self.mode == 'position':
            self.position_command = command
            drive_cmd = -command if self.flip_direction_sign else command
            self._rx_pdo.position_command = int(drive_cmd*(2**32)/(2*math.pi) + self.pos_cmd_offset)
            # print(self.name,' position cmd: ', self._rx_pdo.position_command)

        elif self.mode == 'velocity':
            # clip command, if needed
            v_limit = self.params['motor_limits']['velocity']
            if abs(command) > v_limit:
                print(self.name, ' velocity command of ', command, ' exceeds limit of ', v_limit)
                command = np.sign(command) * v_limit

            self.velocity_command = command
            drive_cmd = -command if self.flip_direction_sign else command
            drive_cmd *= 60/(math.pi*2) # convert from rad/s command to rpm
            self._rx_pdo.velocity_command = int(drive_cmd*1000) # scale velocity command by 1000 and convert to int
            

        elif self.mode == 'torque':
            # clip command, if needed
            t_limit = self.params['motor_limits']['torque']
            if abs(command) > t_limit:
                print(self.name, ' torque command of ', command, ' exceeds limit of ', t_limit)
                command = np.sign(command) * t_limit

            self.torque_command = command
            kt = self.params['motor_params']['kt']
            k_tanh = self.params['motor_params']['k_tanh']
            current_target = np.arctanh(command * k_tanh / kt) / k_tanh

            if self.flip_torque_sign:
                current_target = -current_target
            self._rx_pdo.current_command = int(current_target*1000)
            # print(self.name,' current cmd: ', self._rx_pdo.current_command,', input cmd ',in_cmd, command)

        # command torque (current) feed forward, if provided a target
        if not torque_ff == 0:
            kt = self.params['motor_params']['kt']
            k_tanh = self.params['motor_params']['k_tanh']
            current_ff = np.arctanh(torque_ff * k_tanh / kt) / k_tanh
            if self.flip_torque_sign:
                current_ff *= -1
            self._rx_pdo.torque_ff = int(current_ff/self.params['drive_params']['i_cont']*1000)

class AXON():
    # Mode-change tunables. The position-target path enters velocity mode first,
    # drives position to 0 with a capped velocity command, then flips firmware
    # into position mode once at home.
    BLEND_STEPS = 1000
    ZERO_VELOCITY_KP = 1.0       # rad/s commanded per rad of position error
    ZERO_VELOCITY_CAP = 1.0      # cap on the zeroing velocity command [rad/s]
    AT_REST_VEL_TOL = 0.01       # rad/s — step 0 waits below this
    AT_HOME_POS_TOL = 0.2        # rad   — step 3 hands off below this
    AT_HOME_VEL_TOL = 0.01       # rad/s — step 3 hands off below this
    GAIN_BLEND_DELAY_CYCLES = 100  # cycles to wait after writing mode SDO before blending

    # 0x8002/mode_change_idx values
    OP_MODE_OFF = 0x0000
    OP_MODE_TORQUE_OR_POSITION = 0x0005
    OP_MODE_VELOCITY = 0x0006

    def __init__(self, slave, name, params):

        self._slave = slave
        self.name = name

        print('\n#### Configuring ',self.name,' ####\n')

        self._sdo_queue = queue.Queue()
        self._sdo_busy = False
        self._worker_thread = threading.Thread(target=self._sdo_worker, daemon=True)
        self._worker_thread.start()

        esi_file = params['esi_file_name']
        with open(f"{dyno_paths.dyno_directory}/{esi_file}", 'r', encoding='utf-8') as f:
            self.esi_dict = xmltodict.parse(f.read())

        rxpdo_objects, rxpdo_fxp_dict = self.esi_parser('RxPdo')
        txpdo_objects, txpdo_fxp_dict = self.esi_parser('TxPdo')

        class RxPDO(ctypes.Structure):
            _pack_ = 1
            _fields_ = rxpdo_objects
        self.RxPDO = RxPDO

        class TxPDO(ctypes.Structure):
            _pack_ = 1
            _fields_ = txpdo_objects
        self.TxPDO = TxPDO

        self._rx_pdo = RxPDO()
        self._rx_pdo_tasks = []
        for object in rxpdo_objects:
            object_name = object[0]
            setattr(self, object_name, 0)   
            if object_name in rxpdo_fxp_dict.keys():
                step = rxpdo_fxp_dict[object_name]['step']
                shift = rxpdo_fxp_dict[object_name]['shift']
                def write_task(f=object_name, step=step, shift = shift):
                    value = int((getattr(self, f) - shift) / step)
                    setattr(self._rx_pdo, f, value)
            else:
                def write_task(f=object_name):
                    value = int(getattr(self, f))
                    setattr(self._rx_pdo, f, value)
            self._rx_pdo_tasks.append(write_task)    

        self._tx_pdo = TxPDO()
        self._tx_pdo_tasks = []
        for object in txpdo_objects:
            object_name = object[0]
            setattr(self, object_name, 0)
            if object_name in txpdo_fxp_dict.keys():
                step = txpdo_fxp_dict[object_name]['step']
                shift = txpdo_fxp_dict[object_name]['shift']
                def read_task(f=object_name, step=step, shift = shift):
                    value = getattr(self._tx_pdo, f)*step + shift
                    setattr(self, f, value)
            else:
                def read_task(f=object_name):
                    value = getattr(self._tx_pdo, f)
                    setattr(self, f, value)
            self._tx_pdo_tasks.append(read_task)   

        self.pos_offset = 0
        self.velocity = 0
        self.position = 0
        self.current = 0

        self.shutdown = False

        self.mode_dict = {'position':2,'velocity':1,'torque':0,'off':-1}

        self._slave.sdo_write(index=0x8001, subindex=32, data=struct.pack('<f', 1.396))
        self._slave.sdo_write(index=0x8001, subindex=33, data=struct.pack('<f', -1.396))

        # Extract SDO indexes to use later
        sdo_objects = self.esi_dict['EtherCATInfo']['Descriptions']['Devices']['Device']['Profile']['Dictionary']['DataTypes']['DataType']
        DT8002 = [obj for obj in sdo_objects if obj['Name'] == 'DT8002'][0]['SubItem']
        DT8001 = [obj for obj in sdo_objects if obj['Name'] == 'DT8001'][0]['SubItem']

        self.mode_change_idx = [int(obj['SubIdx']) for obj in DT8002 if obj['Name'] == 'mode'][0]
        self.actuator_type_idx = [int(obj['SubIdx']) for obj in DT8002 if obj['Name'] == 'actuator_type'][0]
        self.saturation_idx = [int(obj['SubIdx']) for obj in DT8001 if obj['Name'] == 'Limits__Motor__Effort__Saturate__Relative_val'][0]
    
        self.actuator_type = slave.sdo_read(0x8002, 2).split(b'\x00')[0].decode('utf-8', 'ignore').strip()
        print('AXON Type Identifier: ', self.actuator_type)

        with open(dyno_paths.dyno_config_directory+'/actuator_library.yaml', 'r') as f:
            actuator_library = yaml.safe_load(f)

        matching_config_entries = [key for key in actuator_library.keys() if self.actuator_type in actuator_library[key]['names']]
        assert len(matching_config_entries) == 1 
        actuator_params = actuator_library[matching_config_entries[0]]

        self.gains = {
            'torque':{'kp':0,'kd':0},
            'velocity':actuator_params['gains']['velocity'],
            'position':actuator_params['gains']['position']
        }

        # Read actuator limits
        lower_torque_limit_idx = [int(obj['SubIdx']) for obj in DT8001 if obj['Name'] == 'Limits__Actuator__Effort__Lower_nm'][0]
        upper_torque_limit_idx = [int(obj['SubIdx']) for obj in DT8001 if obj['Name'] == 'Limits__Actuator__Effort__Upper_nm'][0]
        lower_velocity_limit_idx = [int(obj['SubIdx']) for obj in DT8001 if obj['Name'] == 'Limits__Actuator__Velocity__Lower_radps'][0]
        upper_velocity_limit_idx = [int(obj['SubIdx']) for obj in DT8001 if obj['Name'] == 'Limits__Actuator__Velocity__Upper_radps'][0]

        self.reference_saturation_level =  ctypes.c_float.from_buffer_copy(self._slave.sdo_read(index=0x8001, subindex=self.saturation_idx, size=4)).value
        self.position_slew_saturation = min(1, abs(actuator_params['slew_saturation']))

        self.torque_limit = min(
            abs(ctypes.c_float.from_buffer_copy(self._slave.sdo_read(index=0x8001, subindex=lower_torque_limit_idx, size=4)).value),
            abs(ctypes.c_float.from_buffer_copy(self._slave.sdo_read(index=0x8001, subindex=upper_torque_limit_idx, size=4)).value),
            abs(actuator_params['limits']['torque_nm']))
        self.velocity_limit = min(
            abs(ctypes.c_float.from_buffer_copy(self._slave.sdo_read(index=0x8001, subindex=lower_velocity_limit_idx, size=4)).value),
            abs(ctypes.c_float.from_buffer_copy(self._slave.sdo_read(index=0x8001, subindex=upper_velocity_limit_idx, size=4)).value),
            abs(actuator_params['limits']['velocity_radps']))
        self.torque_limit *= min(1, max(0, self.reference_saturation_level)) 

        print('\nAXON self reported limits:')
        print('\tTorque: ',self.torque_limit,' Nm')
        print('\tVelocity: ',self.velocity_limit,' rad/s')

        self.state = None
        self.sw_enable = True

        self.mode = 'off'
        self.switching_modes = False

        self.data_counter = 0

        # In absensce of loading params from config, define gear ratio as 1
        self.params = {'gear_ratio':1}

        self.cmd__actuator__effort__nm__fxp = 0
        self.cmd__actuator__position__rad__fxp = 0
        self.cmd__actuator__velocity__radps__fxp = 0
        self.cmd_motor_effort_a = 0
        self.cmd_motor_position_rad = 0
        self.cmd_motor_velocity_radps = 0
        self.gain__actuator_impedance_kd__nmsprad__fxp = 0
        self.gain__actuator_impedance_kp__nmprad__fxp = 0

        self.torque_command = math.nan
        self.velocity_command = math.nan
        self.position_command = math.nan

        self.turn_offset = 0

        # Start at safe values
        self.fault = True
        self.enabled = False

        print(self.name, '  ', self.mode)

    def _sdo_worker(self):
        """ Background loop that executes blocking SDO writes """
        while True:
            # This blocks the background thread, but NOT the real-time loop
            index, subindex, data = self._sdo_queue.get()
            self._sdo_busy = True
            try:
                self._slave.sdo_write(index, subindex, data, release_gil=True)
            except Exception as e:
                print(f"[{self.name}] Background SDO Error: {e}")
            finally:
                self._sdo_busy = False
                self._sdo_queue.task_done()

    def esi_parser(self, pdo_type):
        assert pdo_type in ['RxPdo', 'TxPdo']

        pdo_entries = self.esi_dict['EtherCATInfo']['Descriptions']['Devices']['Device'][pdo_type]['Entry']

        data_types = self.esi_dict['EtherCATInfo']['Descriptions']['Devices']['Device']['Profile']['Dictionary']['DataTypes']['DataType']
        data_types = [x for x in data_types if x['Name'] == 'DT8002'][0]['SubItem']
        data_types = {data_type['Name']:data_type for data_type in data_types}
    
        fxp_dict = {}

        print('Mapped ',pdo_type,' entries:')

        struct_entries = []
        for entry in pdo_entries:
            if entry['DataType'] == 'INT' and entry['BitLen'] == '16':
                struct_entries.append((entry['Name'], ctypes.c_int16))
            elif entry['DataType'] == 'UINT' and entry['BitLen'] == '16':
                struct_entries.append((entry['Name'], ctypes.c_uint16))
            elif entry['DataType'] == 'REAL' and entry['BitLen'] == '32':
                struct_entries.append((entry['Name'], ctypes.c_float))
            elif entry['DataType'] == 'UDINT' and entry['BitLen'] == '32':
                struct_entries.append((entry['Name'], ctypes.c_uint32))
            else:
                raise ValueError('PDO bit length of '+entry['BitLen']+' and type of '+entry['DataType']+' is unsupported')
            
            print('\t',entry['Name'])

            if entry['Name'][-3:] == 'fxp':
                lookup_string = 'fixed_point__'+entry['Name']+'__shift'
                sub_idx = int(data_types[lookup_string]['SubIdx'])
                print('lookup ',lookup_string,' subidx ',sub_idx)
                shift = ctypes.c_float.from_buffer_copy(self._slave.sdo_read(0x8002, sub_idx, size=4))

                lookup_string = 'fixed_point__'+entry['Name']+'__step'
                sub_idx = int(data_types[lookup_string]['SubIdx'])
                print('lookup ',lookup_string,' subidx ',sub_idx)
                step = ctypes.c_float.from_buffer_copy(self._slave.sdo_read(0x8002, sub_idx, size=4))

                fxp_dict[entry['Name']] = {'shift': shift.value,
                                            'step': step.value}

                print(entry['Name'],step.value, shift.value)

        return struct_entries, fxp_dict

    def manage_mode_change(self):
        if not self.switching_modes:
            return

        if self.mode_change_step == 0:
            self._mode_change_wait_for_rest()
        elif self.mode_change_step == 1:
            self._mode_change_write_op_mode_sdo()
        elif self.mode_change_step == 2:
            if self.data_counter > (self.mode_change_start + self.GAIN_BLEND_DELAY_CYCLES):
                self._mode_change_blend_gains()
        elif self.mode_change_step == 3:
            self._mode_change_zero_or_finalize()

    # ---- mode-change helpers ----

    @property
    def _blend_gain_mode(self):
        # Position target uses velocity-mode gains during the zeroing phase.
        return 'velocity' if self.target_mode == 'position' else self.target_mode

    def _zeroing_velocity_cmd(self):
        # Capped P-controller on position error; drives position toward 0.
        cmd = -self.ZERO_VELOCITY_KP * self.position
        return max(-self.ZERO_VELOCITY_CAP, min(self.ZERO_VELOCITY_CAP, cmd))

    def _at_home(self):
        return abs(self.position) < self.AT_HOME_POS_TOL and abs(self.velocity) < self.AT_HOME_VEL_TOL

    def _queue_op_mode(self, mode_word):
        self._sdo_queue.put((0x8002, self.mode_change_idx, mode_word.to_bytes(2, 'little')))

    def _mode_change_wait_for_rest(self):
        # Step 0: wait until the actuator stops moving before we start changing anything.
        if abs(self.velocity) < self.AT_REST_VEL_TOL:
            self.mode_change_step = 1
            print(f'[{self.name} mode-change] step 0->1: system at rest '
                  f'(vel={self.velocity:+.4f} rad/s, position={self.position:+.4f} rad). '
                  f'Proceeding to write {self.target_mode}-mode SDO.')

    def _mode_change_write_op_mode_sdo(self):
        # Step 1: send the message that tells the actuator "switch to this control mode now" (velocity or torque).
        if self.target_mode in ('velocity', 'position'):
            self._queue_op_mode(self.OP_MODE_VELOCITY)
            if self.target_mode == 'position':
                print(f'[{self.name} mode-change] step 1->2: queued velocity-mode SDO (0x0006) '
                      f'to drive position->0 capped at {self.ZERO_VELOCITY_CAP:.2f} rad/s '
                      f'(target={self.target_mode}). Will blend velocity gains over {self.BLEND_STEPS} cycles.')
            else:
                print(f'[{self.name} mode-change] step 1->2: queued velocity-mode SDO (0x0006). '
                      f'Will blend gains up over {self.BLEND_STEPS} cycles.')
        else:  # torque
            self._queue_op_mode(self.OP_MODE_TORQUE_OR_POSITION)
            print(f'[{self.name} mode-change] step 1->2: queued torque-mode SDO (0x0005).')

        self.mode_change_start = self.data_counter
        self.mode_change_step = 2

    def _mode_change_blend_gains(self):
        # Step 2: slowly ramp the control gains from 0 up to their target values, so the actuator doesn't lurch when it wakes up.
        gain_mode = self._blend_gain_mode

        if self.target_mode == 'position':
            self.cmd__actuator__velocity__radps__fxp = self._zeroing_velocity_cmd()

        if self.gain__actuator_impedance_kd__nmsprad__fxp < self.gains[gain_mode]['kd']:
            self.gain__actuator_impedance_kd__nmsprad__fxp += self.gains[gain_mode]['kd'] / self.BLEND_STEPS
        if self.gain__actuator_impedance_kp__nmprad__fxp < self.gains[gain_mode]['kp']:
            self.gain__actuator_impedance_kp__nmprad__fxp += self.gains[gain_mode]['kp'] / self.BLEND_STEPS

        print(f"[{self.name} mode-change] step 2: blending {gain_mode} kp={self.gain__actuator_impedance_kp__nmprad__fxp:.2f}/{self.gains[gain_mode]['kp']:.2f} kd={self.gain__actuator_impedance_kd__nmsprad__fxp:.3f}/{self.gains[gain_mode]['kd']:.3f}", end='\r')

        if (self.gain__actuator_impedance_kd__nmsprad__fxp >= self.gains[gain_mode]['kd'] and
            self.gain__actuator_impedance_kp__nmprad__fxp >= self.gains[gain_mode]['kp']):
            self.mode_change_step = 3
            print(f'[{self.name} mode-change] step 2->3: gains fully blended in.')

    def _mode_change_zero_or_finalize(self):
        # Step 3: if heading to position mode, keep nudging position toward 0; once at 0, hand off to position mode. Otherwise we're done.
        if self.target_mode != 'position':
            self._finalize_mode_change()
            print(f'[{self.name} mode-change] step 3: complete -> {self.mode} mode at pos={self.position:+.4f} rad')
            return

        velocity_cmd = self._zeroing_velocity_cmd()
        self.cmd__actuator__velocity__radps__fxp = velocity_cmd

        if self._at_home():
            self._handoff_velocity_to_position()
        else:
            print(f"[{self.name} mode-change] step 3: zeroing p={self.position:+.3f} v={self.velocity:+.3f} vcmd={velocity_cmd:+.3f}", end='\r')

    def _handoff_velocity_to_position(self):
        self._queue_op_mode(self.OP_MODE_TORQUE_OR_POSITION)
        self.cmd__actuator__velocity__radps__fxp = 0
        self.cmd__actuator__position__rad__fxp = 0
        self.gain__actuator_impedance_kd__nmsprad__fxp = self.gains['position']['kd']
        self.gain__actuator_impedance_kp__nmprad__fxp = self.gains['position']['kp']
        self._finalize_mode_change()
        print(f'[{self.name} mode-change] step 3: reached home (pos={self.position:+.4f} rad, vel={self.velocity:+.4f} rad/s) -> position mode')

    def _finalize_mode_change(self):
        self.switching_modes = False
        self.mode = self.target_mode

    # This method intakes a new target control mode
    def command_operating_mode(self, mode):
        assert mode in ['torque', 'velocity', 'position']

        print(f'[{self.name} mode-change] requested: {self.mode} -> {mode} '
              f'(current position={self.position:+.4f} rad, vel={self.velocity:+.4f} rad/s). '
              f'Writing mode-off SDO and zeroing effort/gains; cmd_position held at current position.')
        # Write command to turn off
        self._sdo_queue.put((0x8002, self.mode_change_idx, 0x0000.to_bytes(2, 'little')))
        self.target_mode = mode
        self.switching_modes = True
        self.mode_change_step = 0

        # Zeroing torque and gains should zero out any output torque
        self.gain__actuator_impedance_kd__nmsprad__fxp = 0
        self.gain__actuator_impedance_kp__nmprad__fxp = 0

        # Write safe values for the other commands
        self.cmd__actuator__effort__nm__fxp = 0
        self.cmd__actuator__velocity__radps__fxp = 0
        self.cmd__actuator__position__rad__fxp = self.position

        self.position_command = math.nan
        self.velocity_command = math.nan
        self.torque_command = math.nan
            
    def process_txpdo(self):
        '''Reads the latest input bytes from the slave and populates the TxPDO structure.'''
        if len(self._slave.input) == ctypes.sizeof(self._tx_pdo):
            self._tx_pdo = self.TxPDO.from_buffer_copy(self._slave.input)
        else:
            print(f'WARNING: AXON input buffer size mismatch! Expected {ctypes.sizeof(self._tx_pdo)}, Got {len(self._slave.input)}')

        for task in self._tx_pdo_tasks:
            task()

        if abs(self.position - self.actuator__position__rad) > 1:
            if not self.position == 0:
                while self.position - (self.actuator__position__rad + math.pi*2*self.turn_offset) > 1:
                    self.turn_offset += 1
                while self.position - (self.actuator__position__rad + math.pi*2*self.turn_offset) < -1:
                    self.turn_offset -= 1

        # Write certain topics to common class attributes
        self.position = self.aps__position__rad # + math.pi*2*self.turn_offset
        self.kd = self.gain__actuator_impedance_kd__nmsprad__fxp
        self.kp = self.gain__actuator_impedance_kp__nmprad__fxp
        self.aps_position_fixed_point = self.aps__position__rad
        self.inc_position = self.actuator__position__rad
        self.motor_position = 0
        self.velocity = self.actuator__velocity__radps
        self.effort = self.actuator__effort__nm__fxp
        self.current = 0
        self.servo_drive_mode = self.mode_dict[self.mode]

        self.manage_mode_change()
        self.update_state()

        if self.mode == 'off' and not self.shutdown and self.data_counter > 50:
            self._slave.sdo_write(index=0x8002, subindex=self.mode_change_idx, data=0x0005.to_bytes(2, 'little'))
            self.mode = 'torque'
            print('EtherCAT communication established')
            print('initial mode set to torque')
            print('Mode check: ',self._slave.sdo_read(0x8002, self.mode_change_idx, size=2))

        self.data_counter += 1

    def update_state(self):
        if self.diag__faults__x == 0:
            self.STO = False
            self.disabled = False
        else:
            self.fault = self.diag__faults__x > 3 #Any values greater than 3 correspond to an error keeping the drive from enabling
            self.enabled = (self.diag__faults__x & 2) != 2

    def write_rxpdo(self):
        if self.shutdown and not self.mode == 'off':
            self._slave.sdo_write(index=0x8002, subindex=self.mode_change_idx, data=0x0000.to_bytes(2, 'little'))
            print('Shutdown mode check: ',self._slave.sdo_read(0x8002, self.mode_change_idx, size=2))
            self.mode = 'off'

        for task in self._rx_pdo_tasks:
            task()

        self._slave.output = bytes(self._rx_pdo)

    def send_command(self, command, torque_ff = 0, integrated_position = None):
        if not self.switching_modes:
            if self.mode == 'position':
                self.cmd__actuator__position__rad__fxp = command
                self.cmd__actuator__effort__nm__fxp = torque_ff
                self.position_command = command

            elif self.mode == 'velocity':
                if abs(command) > self.velocity_limit:
                    print(self.name, ' velocity command of ', command, ' exceeds limit of ', self.velocity_limit)
                    command = np.sign(command) * self.velocity_limit
                self.cmd__actuator__velocity__radps__fxp = command 
                self.cmd__actuator__effort__nm__fxp = torque_ff
                self.velocity_command = command

            elif self.mode == 'torque':
                if abs(command) > self.torque_limit:
                    print(self.name, ' torque command of ', command, ' exceeds limit of ', self.torque_limit)
                    command = np.sign(command) * self.torque_limit
                self.cmd__actuator__effort__nm__fxp = command
                self.torque_command = command


class RB430():
    def __init__(self, slave, name, params):

        self._slave = slave
        self.name = name

        print('\n#### Configuring ',self.name,' ####\n')

        esi_file = 'rb430.xml'
        with open(esi_file, 'r', encoding='utf-8') as f:
            self.esi_dict = xmltodict.parse(f.read())

        rxpdo_objects, rxpdo_fxp_dict = self.esi_parser('RxPdo')
        txpdo_objects, txpdo_fxp_dict = self.esi_parser('TxPdo')

        class RxPDO(ctypes.Structure):
            _pack_ = 1
            _fields_ = rxpdo_objects
        self.RxPDO = RxPDO

        class TxPDO(ctypes.Structure):
            _pack_ = 1
            _fields_ = txpdo_objects
        self.TxPDO = TxPDO

        self._rx_pdo = RxPDO()
        self._rx_pdo_tasks = []
        for object in rxpdo_objects:
            object_name = object[0]
            setattr(self, object_name, 0)   
            if object_name in rxpdo_fxp_dict.keys():
                step = rxpdo_fxp_dict[object_name]['step']
                shift = rxpdo_fxp_dict[object_name]['shift']
                def write_task(f=object_name, step=step, shift = shift):
                    value = int((getattr(self, f) - shift) / step)
                    setattr(self._rx_pdo, f, value)
            else:
                def write_task(f=object_name):
                    value = int(getattr(self, f))
                    setattr(self._rx_pdo, f, value)
            self._rx_pdo_tasks.append(write_task)    

        self._tx_pdo = TxPDO()
        self._tx_pdo_tasks = []
        for object in txpdo_objects:
            object_name = object[0]
            setattr(self, object_name, 0)
            if object_name in txpdo_fxp_dict.keys():
                step = txpdo_fxp_dict[object_name]['step']
                shift = txpdo_fxp_dict[object_name]['shift']
                def read_task(f=object_name, step=step, shift = shift):
                    value = getattr(self._tx_pdo, f)*step + shift
                    setattr(self, f, value)
            else:
                def read_task(f=object_name):
                    value = getattr(self._tx_pdo, f)
                    setattr(self, f, value)
            self._tx_pdo_tasks.append(read_task)   

        self.pos_offset = 0
        self.velocity = 0
        self.position = 0
        self.current = 0

        self.velocity_limit = 8.4
        self.torque_limit = 55

        self.shutdown = False

        self.mode_dict = {'position':2,'velocity':1,'torque':0,'off':-1}

        # print('Max cycle rate ', ctypes.c_float.from_buffer_copy(self._slave.sdo_read(0x8002, 30, size=32)))

        self._slave.sdo_write(index=0x8001, subindex=32, data=struct.pack('<f', 1.396))
        self._slave.sdo_write(index=0x8001, subindex=33, data=struct.pack('<f', -1.396))
        # for idx in range(30, 46):
        #     print('0x8001: ',idx,' = ',ctypes.c_float.from_buffer_copy(self._slave.sdo_read(0x8001, idx, size=4)))

        self.gains = {
            'velocity':{
                'kp':8,
                'kd':16
            },
            'position':{
                'kp':250,
                'kd':8
            }
        }

        self.state = None
        self.sw_enable = True

        self.mode = 'off'
        self.switching_modes = False

        self.data_counter = 0

        self.cmd__actuator__effort__nm__fxp = 0
        self.cmd__actuator__position__rad__fxp = 0
        self.cmd__actuator__velocity__radps__fxp = 0
        self.cmd_motor_effort_a = 0
        self.cmd_motor_position_rad = 0
        self.cmd_motor_velocity_radps = 0
        self.gain__actuator_impedance_kd__nmsprad__fxp = 0

        self.params = {'gear_ratio':1}

        self.torque_command = math.nan
        self.velocity_command = math.nan
        self.position_command = math.nan

        self.fault_dict = {
            1: 'ECAT_WATCHDOG',
            2: 'MOTOR_DISABLED',
            4: 'ESTOP',
            8: 'NAN',
            16: 'DRV',
            64: 'CMP_TO',
            128: 'OUT_TO',
            256: 'THERMAL',
            512: 'MOTOR',
            1024: 'ACTUATOR',
            2048: 'JOINT'
        }

    def update_state(self):
        """
        Monitors the diagnostic fault bitmask and tracks the drive state.
        Functions identically to the AKD update_modes() state tracker.
        """
        fault_val = int(self.diag__faults__x)
        
        # In the RB430, bit 2 (value=2) simply means the motor is disabled. 
        # We mask it out so we only trigger a hard 'FAULT' state for real errors.
        active_faults = fault_val & ~2 

        if active_faults > 0:
            if self.state != 'fault':
                # Find all active fault names based on the bitmask
                fault_names = [name for bit, name in self.fault_dict.items() if (fault_val & bit)]
                print(f"{self.name} State: FAULT! ({', '.join(fault_names)})")
            self.state = 'fault'
            
        elif fault_val == 2:
            if self.state != 'disabled':
                print(f"{self.name} State: Disabled")
            self.state = 'disabled'
            
        elif fault_val == 0:
            if self.state != 'enabled':
                print(f"{self.name} State: Enabled")
            self.state = 'enabled'

    def esi_parser(self, pdo_type):
        assert pdo_type in ['RxPdo', 'TxPdo']

        pdo_entries = self.esi_dict['EtherCATInfo']['Descriptions']['Devices']['Device'][pdo_type]['Entry']

        data_types = self.esi_dict['EtherCATInfo']['Descriptions']['Devices']['Device']['Profile']['Dictionary']['DataTypes']['DataType']
        data_types = [x for x in data_types if x['Name'] == 'DT8002'][0]['SubItem']
        data_types = {data_type['Name']:data_type for data_type in data_types}
    
        fxp_dict = {}

        print('Mapped ',pdo_type,' entries:')

        struct_entries = []
        for entry in pdo_entries:
            if entry['DataType'] == 'INT' and entry['BitLen'] == '16':
                struct_entries.append((entry['Name'], ctypes.c_int16))
            elif entry['DataType'] == 'UINT' and entry['BitLen'] == '16':
                struct_entries.append((entry['Name'], ctypes.c_uint16))
            elif entry['DataType'] == 'REAL' and entry['BitLen'] == '32':
                struct_entries.append((entry['Name'], ctypes.c_float))
            elif entry['DataType'] == 'UDINT' and entry['BitLen'] == '32':
                struct_entries.append((entry['Name'], ctypes.c_uint32))
            else:
                raise ValueError('PDO bit length of '+entry['BitLen']+' and type of '+entry['DataType']+' is unsupported')
            
            print('\t',entry['Name'])

            if entry['Name'][-3:] == 'fxp':
                lookup_string = 'fixed_point__'+entry['Name']+'__shift'
                shift = ctypes.c_float.from_buffer_copy(self._slave.sdo_read(0x8002, int(data_types[lookup_string]['SubIdx']), size=4))

                lookup_string = 'fixed_point__'+entry['Name']+'__step'
                step = ctypes.c_float.from_buffer_copy(self._slave.sdo_read(0x8002, int(data_types[lookup_string]['SubIdx']), size=4))

                fxp_dict[entry['Name']] = {'shift': shift.value,
                                            'step': step.value}

                print(entry['Name'],step.value, shift.value)

        return struct_entries, fxp_dict

    def manage_mode_change(self):
        if self.switching_modes:
            # Let system come to a rest
            if self.mode_change_step == 0 and abs(self.velocity) < 0.01:
                if self.target_mode == 'position':
                    self.slew_limit = 0
                    self.mode_change_step = 1
                    print(f'[{self.name} mode-change] step 0->1: system at rest (vel={self.velocity:+.4f} rad/s). '
                          f'Target is position mode -> will slew from current position {self.position:+.4f} rad '
                          f'to home/base_position {self.base_position:+.4f} rad before applying mode.')
                else:
                    self.mode_change_step = 3
                    print(f'[{self.name} mode-change] step 0->3: system at rest (vel={self.velocity:+.4f} rad/s). '
                          f'Target is {self.target_mode} mode -> skipping slew, applying mode directly.')

            # For changing to position, slew the actuator back to it's home range
            elif self.mode_change_step == 1:
                if abs(self.position - self.base_position) < 0.01:
                    self.mode_change_step = 2
                    print(f'[{self.name} mode-change] step 1->2: reached home '
                          f'(position={self.position:+.4f} rad, target={self.base_position:+.4f} rad, '
                          f'err={self.base_position - self.position:+.4f} rad). Waiting for velocity to settle.')
                else:
                    if not self.mode == 'velocity':
                        self._slave.sdo_write(index=0x8002, subindex=103, data=0x0006.to_bytes(2, 'little'))
                        self.gain__actuator_impedance_kd__nmsprad__fxp = self.gains['velocity']['kd']
                        self.gain__actuator_impedance_kp__nmprad__fxp = self.gains['velocity']['kp']
                        self.mode = 'velocity'
                        print(f'[{self.name} mode-change] step 1: switching to velocity mode to drive slew '
                              f'(err to home = {self.base_position - self.position:+.4f} rad).')
                    else:
                        command = 4*(self.base_position - self.position)
                        if abs(command) > self.slew_limit:
                            command = self.slew_limit * (command / abs(command))
                            if self.slew_limit < 4:
                                self.slew_limit += 0.005

                        if self.data_counter % 100 == 0:
                            print(f'[{self.name} mode-change] step 1: slewing to home -> '
                                  f'err={self.base_position - self.position:+.4f} rad, '
                                  f'vel_cmd={command:+.4f} rad/s, slew_limit={self.slew_limit:.3f}, '
                                  f'effort={self.effort:+.3f} Nm')
                        self.cmd__actuator__velocity__radps__fxp = command

            if self.mode_change_step == 2 and abs(self.velocity) < 0.01:
                self.mode_change_step = 3
                print(f'[{self.name} mode-change] step 2->3: velocity settled at home '
                      f'(vel={self.velocity:+.4f} rad/s). Applying target mode.')

            # Once the system has stabilized we'll send the new mode commands
            elif self.mode_change_step == 3:
                if self.target_mode == 'torque':
                    self._slave.sdo_write(index=0x8002, subindex=103, data=0x0005.to_bytes(2, 'little'))
                    self.gain__actuator_impedance_kd__nmsprad__fxp = 0
                    self.gain__actuator_impedance_kp__nmprad__fxp = 0
                    print(f'[{self.name} mode-change] step 3: wrote torque mode (gains zeroed).')
                elif self.target_mode =='velocity':
                    self._slave.sdo_write(index=0x8002, subindex=103, data=0x0006.to_bytes(2, 'little'))
                    self.gain__actuator_impedance_kd__nmsprad__fxp = self.gains['velocity']['kd']
                    self.gain__actuator_impedance_kp__nmprad__fxp = self.gains['velocity']['kp']
                    print(f'[{self.name} mode-change] step 3: wrote velocity mode '
                          f'(kp={self.gains["velocity"]["kp"]}, kd={self.gains["velocity"]["kd"]}).')
                elif self.target_mode =='position':
                    self._slave.sdo_write(index=0x8002, subindex=103, data=0x0005.to_bytes(2, 'little'))
                    self.cmd__actuator__position__rad__fxp = self.base_position
                    self.cmd__actuator__velocity__radps__fxp = 0
                    self.gain__actuator_impedance_kd__nmsprad__fxp = self.gains['position']['kd']
                    self.gain__actuator_impedance_kp__nmprad__fxp = self.gains['position']['kp']
                    print(f'[{self.name} mode-change] step 3: wrote position mode, '
                          f'cmd_position={self.base_position:+.4f} rad '
                          f'(kp={self.gains["position"]["kp"]}, kd={self.gains["position"]["kd"]}).')
                self.mode_write_time = self.data_counter
                self.mode_change_step = 4

            elif self.mode_change_step == 4 and self.data_counter > (self.mode_write_time+100):
                self.switching_modes = False
                self.mode = self.target_mode
                print(f'[{self.name} mode-change] step 4: complete. Now in {self.mode} mode '
                      f'at position {self.position:+.4f} rad.\n')

    # This method intakes a new target control mode
    def command_operating_mode(self, mode, home=0):
        assert mode in ['torque', 'velocity', 'position']
        print(f'[{self.name} mode-change] requested: {self.mode} -> {mode} '
              f'(current position={self.position:+.4f} rad, home/base_position={home:+.4f} rad). '
              f'Zeroing effort/gains and holding current position until slew begins.')
        self.target_mode = mode
        self.switching_modes = True
        self.mode_change_step = 0

        self.base_position = home

        # Zeroing torque and gains should zero out any output torque
        self.cmd__actuator__effort__nm__fxp = 0
        self.gain__actuator_impedance_kd__nmsprad__fxp = 0
        self.gain__actuator_impedance_kp__nmprad__fxp = 0

        # Write safe values for the other commands
        self.cmd__actuator__velocity__radps__fxp = 0
        self.cmd__actuator__position__rad__fxp = self.position

        self.position_command = math.nan
        self.velocity_command = math.nan
        self.torque_command = math.nan
            

    def process_txpdo(self):
        '''Reads the latest input bytes from the slave and populates the TxPDO structure.'''
        if len(self._slave.input) == ctypes.sizeof(self._tx_pdo):
            self._tx_pdo = self.TxPDO.from_buffer_copy(self._slave.input)
        else:
            print(f'WARNING: AKD input buffer size mismatch! Expected {ctypes.sizeof(self._tx_pdo)}, Got {len(self._slave.input)}')

        for task in self._tx_pdo_tasks:
            task()

        # Write certain topics to common class attributes
        self.position = self.actuator__position__rad
        self.motor_position = self.motor__position__rad
        self.velocity = self.actuator__velocity__radps__fxp
        self.effort = self.actuator__effort__nm__fxp
        self.current = self.motor__effort__a__fxp
        self.servo_drive_mode = self.mode_dict[self.mode]

        self.update_state()

        self.manage_mode_change()

        if self.mode == 'off' and not self.shutdown and self.data_counter > 30:
            self._slave.sdo_write(index=0x8002, subindex=103, data=0x0005.to_bytes(2, 'little'))
            self.mode = 'torque'
            print('initial mode set to torque')
            print('Mode check: ',self._slave.sdo_read(0x8002, 103, size=2))

        self.data_counter += 1

    def write_rxpdo(self):
        if self.shutdown and not self.mode == 'off':
            self._slave.sdo_write(index=0x8002, subindex=103, data=0x0000.to_bytes(2, 'little'))
            self.mode = 'off'

        for task in self._rx_pdo_tasks:
            task()

        self._slave.output = bytes(self._rx_pdo)

    def send_command(self, command, torque_ff = 0, integrated_position = None):
        if not self.switching_modes:
            if self.mode == 'position':
                self.cmd__actuator__position__rad__fxp = command + self.base_position
                # self.cmd__actuator__effort__nm__fxp = torque_ff
                self.cmd__actuator__effort__nm__fxp = 0
                self.position_command = command

            elif self.mode == 'velocity':
                if abs(command) > self.velocity_limit:
                    print(self.name, ' velocity command of ', command, ' exceeds limit of ', self.velocity_limit)
                    command = np.sign(command) * self.velocity_limit
                self.cmd__actuator__velocity__radps__fxp = command 
                # self.cmd__actuator__effort__nm__fxp = torque_ff
                self.cmd__actuator__effort__nm__fxp = 0
                self.velocity_command = command

            elif self.mode == 'torque':
                if abs(command) > self.torque_limit:
                    print(self.name, ' torque command of ', command, ' exceeds limit of ', self.torque_limit)
                    command = np.sign(command) * self.torque_limit
                self.cmd__actuator__effort__nm__fxp = command
                self.torque_command = command


# Each device needs an entry in this dictionary for the controller to properly laod it.
DEVICE_CLASSES = {
    'EK1100': {
        'id': EK1100_PRODUCT_CODE,
        'class': EK1100
        },
    'EL2002': {
        'id': EL2002_PRODUCT_CODE,
        'class': EL2002
        },
    'EL1002': {
        'id': EL1002_PRODUCT_CODE,
        'class': EL1002
        },
    'EL2502': {
        'id': EL2502_PRODUCT_CODE,
        'class': EL2502,
        'has_dc': False # Bypassed with dummy class
        },
    'EL2004': {
        'id': EL2004_PRODUCT_CODE,
        'class': EL2004,
        'has_dc': False
        },
    'EL2004_12v': {
        'id': EL2024_PRODUCT_CODE, 
        'class': EL2004,
        'has_dc': False
        },
    'EL3208': {
        'id': EL3208_PRODUCT_CODE,
        'class': EL3208,
        'has_dc': False
        },
    'EL5042': {
        'id': EL5042_PRODUCT_CODE,
        'class': EL5042,
        'has_dc': True
        },
    'ELM3002': {
        'id': ELM3002_PRODUCT_CODE,
        'class': ELM3002
        },
    'ELM3004': {
        'id': ELM3004_PRODUCT_CODE,
        'class': ELM3004,
        'has_dc': False
        },
    'AKD': {
        'id': AKD_PRODUCT_CODE,
        'class': AKD
        },
    'AXON': {
        'id': AXON_PRODUCT_CODE,
        'class': AXON
        },
    'RB430': {
        'id': AXON_PRODUCT_CODE,
        'class': RB430
        }
    }
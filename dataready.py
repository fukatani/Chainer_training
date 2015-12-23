import os
import spidev
import time
import RPi.GPIO as GPIO
from optparse import OptionParser

spi = None
write_file = None

def callback(pin):
    time.sleep(0.00001)
    _, resp_upper = spi.xfer2([0x83,0x00])
    time.sleep(0.00001)
    _, resp_lower = spi.xfer2([0x82,0x00])
    resp = resp_upper * 2 ** 8 + resp_lower
    print(resp)
    global write_file
    write_file.write(str(resp) + '\n')

def callback_adr(pin):
    time.sleep(0.00001)
    _, resp = spi.xfer2([0xa5,0x00])
    print(str(resp)+'@0x25')

def int_communication(freq, single, adr_change):
    GPIO_DR = 6
    i = 0
    while os.path.exists('data_' + str(i) + '.dat'):
        i += 1
    global write_file
    write_file = open('data_' + str(i) + '.dat', 'w')
    global spi
    spi = spidev.SpiDev()
    spi.open(0,0)
    spi.max_speed_hz=(freq)
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(GPIO_DR, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
    GPIO.add_event_detect(GPIO_DR, GPIO.RISING)
    if not adr_change:
        GPIO.add_event_callback(GPIO_DR, callback)
    else:
        GPIO.add_event_callback(GPIO_DR, callback_adr)
    if single:
        if not adr_change:
            callback(6)
        else:
            callback_adr(6)
    else:
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            GPIO.cleanup()
            #global write_file
    time.sleep(0.1)
    write_file.close()
    print('Written to data_' + str(i) + '.dat')

if __name__ == '__main__':
    optparser = OptionParser()
    optparser.add_option("-f","--freq",dest="spi_freq",
                         default=1000000, help="Spi frequency, Default=1000000")
    optparser.add_option("-c","--clean",dest="clean_flag", action="store_true",
                         default=False, help="Clean previous data or not, Default=False")
    optparser.add_option("-n","--single",dest="single", action="store_true",
                         default=False, help="Not wait data ready and execute single reading, Default=False")
    optparser.add_option("-a","--adr",dest="adr_change", action="store_true",
                         default=False, help="Not wait data ready and execute single reading, Default=False")
    (options, args) = optparser.parse_args()
    if options.clean_flag and os.path.exists('./data_0.dat'):
        os.remove('data_0.dat')
    int_communication(options.spi_freq, options.single, options.adr_change)

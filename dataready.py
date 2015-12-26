import os
import time
from optparse import OptionParser
import data_manager

spi = None
write_file = None

def callback(pin):
    #time.sleep(0.000001)
    _, resp_lower, resp_upper = spi.xfer2([0x82,0x00,0x00])
    resp = resp_upper * 2 ** 8 + resp_lower
    print(resp)
    global write_file
    write_file.write(str(resp) + '\n')

def callback_adr(pin):
    time.sleep(0.00001)
    _, resp = spi.xfer2([0xa5,0x00])
    print(str(resp)+'@0x25')

def int_communication(data_dir, freq, single, adr_change, backend, skip_meas, clk_enable):
    GPIO_DR = 6
    i = 0
    while os.path.exists(os.path.join(data_dir, 'data_' + str(i) + '.dat')):
        i += 1
    global write_file
    if not skip_meas:
        import te_setup
        import RPi.GPIO as GPIO
        import spidev

        spi = spidev.SpiDev()
        write_file = open(os.path.join(data_dir, 'data_' + str(i) + '.dat'), 'w')
        global spi
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

    if backend == 'display_graph':
        data_manager.data_manager(directory=data_dir,
                     data_size=1000,
                     train_size=100,
                     split_mode=False,
                     attenate_flag=False,
                     save_as_png=False).plot()

if __name__ == '__main__':
    optparser = OptionParser()
    optparser.add_option("-f","--freq",dest="spi_freq",
                         default=1000000, help="Spi frequency, Default=1000000")
    optparser.add_option("-c","--clean",dest="clean_flag", action="store_true",
                         default=False, help="Clean previous data or not, Default=False")
    optparser.add_option("-n","--single",dest="single", action="store_true",
                         default=False, help="Not wait data ready and execute single reading, Default=False")
    optparser.add_option("-a","--adr",dest="adr_change", action="store_true",
                         default=False, help="Data read from 0x25, Default=False")
    optparser.add_option("-b","--backend",dest="backend",
                         default=False, help="Backend, Default=False")
    optparser.add_option("-w","--without_setup",dest="without_setup", action="store_true",
                         default=False, help="Without setup, Default=False")
    optparser.add_option("-s","--skip_meas",dest="skip_meas", action="store_true",
                         default=False, help="Skip measurement, Default=False")
    optparser.add_option("-C","--clock_enable",dest="clock_enable", action="store_true",
                         default=False, help="Clk enable during measurement, Default=False")
    (options, args) = optparser.parse_args()

    DATA_DIR = './data/'

    if options.clean_flag and os.path.exists(os.path.join(DATA_DIR, 'data_0.dat')):
        os.remove(os.path.join(DATA_DIR, 'data_0.dat'))
    if options.without_setup:
        te_setup.setup()
    int_communication(DATA_DIR,
                      options.spi_freq,
                      options.single,
                      options.adr_change,
                      options.backend,
                      options.skip_meas,
                      options.clk_enable)

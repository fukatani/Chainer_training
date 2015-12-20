import spidev
import time
import RPIO as GPIO

def write_data(adr, data, spi, wait=0.05):
    spi.xfer2([adr, data])
    time.sleep(wait)

def read_data(adr, data, spi, wait=0.05):
    return spi.xfer2([adr + 0x80, 0x00])

def write_and_readback(adr, data, spi, wait=0.05):
    write_data(adr, data, spi)
    assert read_data(adr, 0x00, spi)[1] == data, 'Read back is failed!'
    print('Succeeded to write ' + str(data) + ' @' + hex(adr))

def setup():
    spi = spidev.SpiDev()
    spi.open(0,0)
    spi.max_speed_hz=(1000000)
    RPIO.setup(5, GPIO.ALT0)

    write_and_readback(0x1c,0x01,spi)
    write_and_readback(0x20,0x01,spi)
    write_and_readback(0x23,0x01,spi)
    write_and_readback(0x25,0x10,spi)
    write_and_readback(0x41,0x01,spi)
    write_and_readback(0x24,0x01,spi)

if __name__ == '__main__':
    setup()
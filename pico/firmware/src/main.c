/* Pico 2 W firmware: Stream AC(lambda) served over UART0 (GP0 = TX, GP1 = RX).
 * The Pico SDK is only used to boot the chip and drive the UART; the agent is plain C. */
#include "hardware/gpio.h"
#include "hardware/uart.h"
#include "pico/time.h"

#include "protocol.h"

#ifndef SAC_BAUD
#define SAC_BAUD 921600
#endif

#define UART uart0
#define TX_PIN 0
#define RX_PIN 1

uint32_t proto_time_us(void) { return time_us_32(); }

static uint8_t rx[PROTO_MAX_PAYLOAD], tx[4 + PROTO_MAX_PAYLOAD];

static uint8_t read_byte(void) { return (uint8_t)uart_getc(UART); }

int main(void) {
    uart_init(UART, SAC_BAUD);
    gpio_set_function(TX_PIN, GPIO_FUNC_UART);
    gpio_set_function(RX_PIN, GPIO_FUNC_UART);
    uart_set_format(UART, 8, 1, UART_PARITY_NONE);
    uart_set_hw_flow(UART, false, false);

    for (;;) {
        while (read_byte() != PROTO_REQ_MAGIC) {
        } /* resynchronize on the frame start */
        uint8_t cmd = read_byte();
        uint32_t len = read_byte();
        len |= (uint32_t)read_byte() << 8;

        uint8_t status;
        uint32_t out_len = 0;
        if (len > PROTO_MAX_PAYLOAD) {
            for (uint32_t i = 0; i < len; i++) read_byte(); /* drain */
            status = PROTO_ERR_LEN;
        } else {
            for (uint32_t i = 0; i < len; i++) rx[i] = read_byte();
            status = proto_handle(cmd, rx, len, tx + 4, &out_len);
        }
        tx[0] = PROTO_RESP_MAGIC;
        tx[1] = status;
        tx[2] = (uint8_t)out_len;
        tx[3] = (uint8_t)(out_len >> 8);
        uart_write_blocking(UART, tx, 4 + out_len);
    }
}

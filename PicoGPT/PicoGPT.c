#include <stdio.h>
#include "pico/stdlib.h"
#include "hardware/timer.h"
#include "hardware/clocks.h"

#include "attention_test.h"

int64_t alarm_callback(alarm_id_t id, void *user_data) {
    // Put your timeout handler code in here
    return 0;
}




int main()
{
    stdio_init_all();
    sleep_ms(100);

    // Timer example code - This example fires off the callback after 2000ms
    //add_alarm_in_ms(2000, alarm_callback, NULL, false);
    // For more examples of timer use see https://github.com/raspberrypi/pico-examples/tree/master/timer
    while (true) {

    printf("System Clock Frequency is %d Hz\n", clock_get_hz(clk_sys));
    printf("USB Clock Frequency is %d Hz\n", clock_get_hz(clk_usb));
    // For more examples of clocks use see https://github.com/raspberrypi/pico-examples/tree/master/clocks

    // 32 ms per attention head
    // That's Softmax(QKT)/rootd*V
    // with residual skip
    // and fully connected
    // actually 2 fullys
    // no layernorm yet
    // but some of this is in floateger

    /* whole model is
     * Layernorm
     * Attention     |
     * Residual      |
     * MLP           |
     * Residual      |
     * Layernorm
     * Attention
     * Layernorm
     * MLP
     * Layernorm
     * Fully Connected to convert to token space
     * Softmax
     * Probabilistic sampling thingie
     * 
     * The highlighted part takes ~32ms
     * Double that, since we are doing two heads. Add a fudge factor since more layernorm
     * Even with float stuff, this is not too slow
     * That's maybe 80ms per token, call it 100ms for the extra stuff
     * So 10 tok/s
     * That is so easily fast enough. That's even without any fancy nonsense
     * Of course, question of accuracy, and I do need to perform _some_ quantisation
     * But this is very feasible I think
     */
    attention_test();

    
        printf("Hello, world!\n");
        sleep_ms(5000);
    }
}

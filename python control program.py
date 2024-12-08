from bleak import BleakClient

HM10_ADDRESS = "60:B6:E1:E1:C6:A6"
UART_CHARACTERISTIC_UUID = "0000ffe1-0000-1000-8000-00805f9b34fb"

async def send_data_loop():
    async with BleakClient(HM10_ADDRESS) as client:
        print("Connected to HM-10 successfully")

        while True:
            # Prompt user for input
            message = input("Enter the message to send (type 'esc' to exit): ")
            
            # Check for the termination keyword
            if message.strip().lower() == "esc":
                print("Termination keyword received. Exiting...")
                break

            # Send the user input as a command
            command = message.encode('utf-8')
            await client.write_gatt_char(UART_CHARACTERISTIC_UUID, command)
            print(f"Command sent successfully: {message}")

# Run the asynchronous task
import asyncio
asyncio.run(send_data_loop())

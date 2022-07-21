# Import required modules and packages
import redis


# Main function
def main():
    # Initialize redis client
    redis_client = redis.Redis(host='127.0.0.1')

    while True:
        stream_num = input("Please Enter Stream Number: ")

        redis_client.xadd(name="Stream_Change",
                          fields={
                              "Stream_Num": stream_num,
                          },
                          maxlen=10,
                          approximate=False)
        redis_client.execute_command(f'XTRIM Stream_Change MAXLEN 10')

        print("Stream Number Sent!")


if __name__ == '__main__':
    main()
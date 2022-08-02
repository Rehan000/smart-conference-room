import redis


# Main function
def main():
    # Initialize redis client
    redis_client = redis.Redis(host='127.0.0.1')

    while True:
        process_name = input("Please Enter Process Name: ")

        redis_client.xadd(name="Process_Change",
                          fields={
                              "Process": process_name,
                          },
                          maxlen=10,
                          approximate=False)
        redis_client.execute_command(f'XTRIM Process_Change MAXLEN 10')

        print("Process Name Sent!")


if __name__ == '__main__':
    main()
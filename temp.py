for index in range(512):

    width = 227
    height = 227
    k_size = 3
    n = 64 * width * height

    w_out = index % width
    h_index = index / height
    h_out = h_index % height
    channel_in = h_index / height
    channel_out = channel_in * k_size ^ 2

    temp = (h_out * w_out) * ((n / height / width) * k_size * k_size) + channel_out
    temp = (channel_out * height + h_out) * width + w_out
    print temp
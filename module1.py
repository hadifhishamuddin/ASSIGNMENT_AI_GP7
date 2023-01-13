def pack_items(items, max_weight):
    # base case: if no items, return empty list and 0 weight
    if not items:
        return [], 0

    # get the first item and its weight
    item = items[0]
    weight = item[1]

    # if the first item is too heavy, skip it
    if weight > max_weight:
        return pack_items(items[1:], max_weight)

    # otherwise, try packing the rest of the items both with and without the current item
    with_item, with_weight = pack_items(items[1:], max_weight - weight)
    without_item, without_weight = pack_items(items[1:], max_weight)

    # return the solution with the greatest value
    if with_weight + item[0] > without_weight:
        return [item] + with_item, with_weight + item[0]
    else:
        return without_item, without_weight

# test the function
items = [(4, 12), (2, 1), (6, 4), (1, 1), (2, 2)]
max_weight = 15
packed_items, total_weight = pack_items(items, max_weight)
print(packed_items)
print(total_weight)
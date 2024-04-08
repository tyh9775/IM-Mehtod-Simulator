# Define the number of lists you want to create
num_lists = 4

# Define a dictionary to store the lists
lists_dict = {}

# Create empty lists and store them in the dictionary
for i in range(1, num_lists + 1):
    list_name = f'list{i}'
    lists_dict[list_name] = []

lists_dict["list1"].append(1)

# Access the lists by their names
print(lists_dict['list1'])  # Output: []
print(lists_dict['list2'])  # Output: []
# and so on...

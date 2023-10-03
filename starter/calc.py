import math

# Define the side length of the equilateral triangle
side_length = 1.0  # You can adjust this value as needed

# Calculate the height of the equilateral triangle
height = math.sqrt(3) / 2 * side_length

# Define the coordinates of the vertices
# Vertex 0 (On the xz plane)
vertex0 = (0, 0, 0)

# Vertex 1 (On the xz plane)
vertex1 = (side_length, 0, 0)

# Vertex 2 (On the xz plane)
vertex2 = (side_length / 2, 0, side_length * math.sqrt(3) / 2)

# Vertex 3 (Top vertex)
vertex3 = (side_length / 2, height, side_length * math.sqrt(3) / 2)

# Now, you have the coordinates of the vertices of the tetrahedron.
# Vertices 0, 1, and 2 are on the xz plane, and the other vertex creates an equilateral triangle above them.

print("Vertex 0:", vertex0)
print("Vertex 1:", vertex1)
print("Vertex 2:", vertex2)
print("Vertex 3:", vertex3)
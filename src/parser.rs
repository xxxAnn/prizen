const NEW_LAYER: usize = 0b00000001;
const NEW_NODE:  usize = 0b00000010;
const META:      usize = 0b00000011;

// Structure
// NEW_LAYER number_of_nodes (NEW_NODE number_of_weights (weight)*number_of_weights bias)*number_of_nodes
// META number_of_input_nodes cst_size cst alpha
// EXP number_of_points ((value)*number_of_input_nodes)*number_of_points (value)*number_of_points
// all vars are float: [u8; 4]
// (x)*b means (x, x, x, ...) b times.

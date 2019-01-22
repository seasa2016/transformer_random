this file tipically modify from the opennmt-py file

training part can work now but leave error checking yet




now we need to turn to use the pretrain model
for pretrain model since the output from decoder should be fix dim,
thereform we can check if we use 
1.same dim
2.use a linear layer to decrease the dim
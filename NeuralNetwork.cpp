#include <iostream>
#include <vector>
#include <math.h>
#include <fstream>
#include <algorithm>


void tokenize(std::string const& str, const char delim,
    std::vector<std::string>& out)
{
    size_t start;
    size_t end = 0;

    while ((start = str.find_first_not_of(delim, end)) != std::string::npos)
    {
        end = str.find(delim, start);
        out.push_back(str.substr(start, end - start));
    }
}


void get_mnist(std::vector<std::vector<float>>& images, std::vector<std::vector<float>>& labels) {
    const std::string inputFile = "C:\\Users\\morit\\Desktop\\CODE\\PYTHON\\Neural_Network\\Build 6\\Build 6.0\\mnist_train.csv";
    std::ifstream inFile(inputFile, std::ios_base::binary);

    inFile.seekg(0, std::ios_base::end);
    size_t length = inFile.tellg();
    inFile.seekg(0, std::ios_base::beg);

    std::vector<char> buffer;
    buffer.reserve(length);
    std::copy(std::istreambuf_iterator<char>(inFile),
        std::istreambuf_iterator<char>(),
        std::back_inserter(buffer));

    /// 28*28 -> 784



    std::string s(buffer.begin(), buffer.end());


    const char delim = '\n';

    std::vector<std::string> out;
    tokenize(s, delim, out);

    //std::vector<std::vector<int>> out2;

    for (int i = 0; i < out.size(); i++) {
        std::vector<std::string> new_vec;
        std::vector<float> true_vec;
        tokenize(out[i], ',', new_vec);

        for (std::string str : new_vec) {
            true_vec.push_back(((float)(atoi(str.c_str()))) / 255.0);
        }

        std::vector<float> label_vec(10, 0);
        label_vec[true_vec[0] * 255] = 1;
        labels.push_back(label_vec);
        true_vec.erase(true_vec.begin());
        images.push_back(true_vec);
    }

}





class NeuralNetwork {

public:
    float e = 2.71828;
    std::vector<int> networkSize;
    float learning_rate = 0.1;
    int number_of_inputs;
    int number_of_layers;
    int number_of_layers_minus_1;
    std::vector<int> layer_neuron_numbers;
    std::vector<std::vector<std::vector<float>>> network;
    std::vector<float> output;

    NeuralNetwork(std::vector<int> _networkSize) {
        networkSize = _networkSize;

        number_of_inputs = networkSize[0];
        number_of_layers = networkSize.size() - 1;
        number_of_layers_minus_1 = number_of_layers - 1;

        std::vector<int> _layer_neuron_numbers(networkSize.begin() + 1, networkSize.end());
        layer_neuron_numbers = _layer_neuron_numbers;

        generate_network();

        //print_network();

    }


    float activation_function(float inp) {
        return 1 / (1 + exp(-inp));
    }

    float d_activation_function(float inp) {
        return inp * (1 - inp);
    }

    void generate_network() {
        int current_number_of_inputs = number_of_inputs;
        int number_of_neurons_in_layer;
        float weight;
        //std::vector<std::vector<float>> layer;
        //std::vector<float> neuron;

        for (int i = 0; i < layer_neuron_numbers.size(); i++) {
            number_of_neurons_in_layer = layer_neuron_numbers[i];
            std::vector<std::vector<float>> layer;

            for (int o = 0; o < number_of_neurons_in_layer; o++) {
                std::vector<float> neuron;

                for (int p = 0; p < (current_number_of_inputs + 1); p++) {
                    weight = rand() / (RAND_MAX + 1.) - 0.5;
                    neuron.push_back(weight);
                }
                layer.push_back(neuron);
            }
            network.push_back(layer);
            current_number_of_inputs = number_of_neurons_in_layer;
        }
    }

    void feed_forward(std::vector<float> inputs) {
        for (std::vector<std::vector<float>> layer : network) {
            std::vector<float> layer_output;

            for (std::vector<float> neuron : layer) {
                float activation = 0;
                for (int i = 0; i < inputs.size(); i++) {
                    //std::cout << neuron[i] << " ";
                    activation += neuron[i] * inputs[i];
                }
                float bias = neuron[neuron.size() - 1];
                activation = activation_function(activation + bias);
                layer_output.push_back(activation);
            }

            inputs = layer_output;
            layer_output.clear();
        }
        output = inputs;
    }

    void train_batch(std::vector<std::vector<float>>& training_inputs, std::vector<std::vector<float>>& targets) {
        std::vector<std::vector<std::vector<float>>> current_network = network;
        for (int training_example_number = 0; training_example_number < training_inputs.size(); training_example_number++) {
            //std::cout << training_inputs.size() << "  " << targets.size();
            std::vector<float> input = training_inputs[training_example_number];
            std::vector<float> target = targets[training_example_number];


            //FORWARD PASS TO GATHER INFO
            std::vector<std::vector<float>> all_layer_outputs;
            std::vector<std::vector<float>> all_layer_inputs;


            for (std::vector<std::vector<float>> layer : network) {
                all_layer_inputs.push_back(input);
                std::vector<float> layer_output;

                for (std::vector<float> neuron : layer) {
                    float activation = 0;
                    for (int i = 0; i < input.size(); i++) {
                        //std::cout << neuron[i] << " ";
                        activation += neuron[i] * input[i];
                    }
                    float bias = neuron[neuron.size() - 1];
                    activation = activation_function(activation + bias);
                    layer_output.push_back(activation);
                }

                input = layer_output;
                all_layer_outputs.push_back(layer_output);
                layer_output.clear();
            }
            output = input;


            // OUTPUT NEURON DELTAS
            std::vector<std::vector<float>> all_neuron_deltas;
            std::vector<float> first_layer_delta;

            for (int i = 0; i < output.size(); i++) {
                float _target = target[i];
                float _output = output[i];

                first_layer_delta.push_back(-(_target - _output) * d_activation_function(_output));
            }
            all_neuron_deltas.push_back(first_layer_delta);

            // HIDDEN DELTAS
            for (int layer_number = 0; layer_number < number_of_layers_minus_1; layer_number++) {
                int real_layer_number = number_of_layers - (layer_number + 2);  
                int num_of_neurons_in_layer = layer_neuron_numbers[real_layer_number];
                std::vector<float> layer_deltas;

                std::vector<float> layer_outputs = all_layer_outputs[real_layer_number];

                for (int a = 0; a < num_of_neurons_in_layer; a++) {
                    float neuron_error = 0;
                    int shallower_layer_number = real_layer_number + 1;
                    int num_of_neurons_in_shallower_layer = layer_neuron_numbers[shallower_layer_number];
                    std::vector<std::vector<float>> shallower_layer = network[shallower_layer_number];

                    for (int b = 0; b < num_of_neurons_in_shallower_layer; b++) {
                        neuron_error += shallower_layer[b][a] * all_neuron_deltas[all_neuron_deltas.size() - 1][b];
                    }

                    layer_deltas.push_back(neuron_error * d_activation_function(layer_outputs[a]));
                }

                all_neuron_deltas.push_back(layer_deltas);
            }

            // UPDATE NEURON WEIGHTS
            for (int layer_number = 0; layer_number < number_of_layers; layer_number++) {
                int real_layer_number = number_of_layers_minus_1 - layer_number;
                int num_of_neurons_in_layer = layer_neuron_numbers[real_layer_number];
                //std::vector<float> layer_inputs = all_layer_inputs[real_layer_number];
                for (int neuron_number = 0; neuron_number < num_of_neurons_in_layer; neuron_number++) {
                    int network_access_index = real_layer_number;

                    for (int weight_number = 0; weight_number < networkSize[network_access_index]; weight_number++) {
                        float weight_error = all_neuron_deltas[layer_number][neuron_number] * all_layer_inputs[real_layer_number][weight_number];
                        current_network[real_layer_number][neuron_number][weight_number] -= weight_error * learning_rate;
                    }
                    current_network[real_layer_number][neuron_number].back() -= all_neuron_deltas[layer_number][neuron_number] * learning_rate;
                }
            }
        }
        network = current_network;
    }


    void save_network(std::string name) {
        std::ofstream out(name);
        for (auto& layer : network)
        {
            //std::vector<std::vector<float>>
            for (auto& neuron : layer)
            {
                //std::vector<float>
                for (auto weight : neuron)
                {
                    out << weight << ',';
                }
                out << '\t';
            }
            out << '\n';
        }
    }


    void load_network(const std::string inputFile) { //BIAS BIAS

        std::ifstream inFile("C:\\Users\\morit\\source\\repos\\NeuralNetwork\\NeuralNetwork\\neural_savefile.txt", std::ios_base::binary);

        inFile.seekg(0, std::ios_base::end);
        size_t length = inFile.tellg();
        inFile.seekg(0, std::ios_base::beg);

        std::vector<char> buffer;
        buffer.reserve(length);
        std::copy(std::istreambuf_iterator<char>(inFile),
            std::istreambuf_iterator<char>(),
            std::back_inserter(buffer));

        std::string s(buffer.begin(), buffer.end());

        //tokenize(s, delim, out);
        std::vector<std::string> layers;
        tokenize(s, '\n', layers);


        network.clear();
        //std::vector<std::vector<std::vector<float>>> network;

        std::vector<std::vector<std::string>> neurons;

        for (int i = 0; i < number_of_layers; i++) {

            std::vector<std::vector<float>> network_layer;

            std::string layer = layers[i];

            std::vector<std::string> neurons;
            tokenize(layer, '\t', neurons);
            
            for (int o = 0; o < layer_neuron_numbers[i]; o++) {

                std::vector<float > network_neuron;

                std::string neuron = neurons[o];

                std::vector<std::string> weights;
                tokenize(neuron, ',', weights);

                for (int p = 0; p < networkSize[i] + 1; p++) {

                    std::string weight = weights[p];

                    network_neuron.push_back(std::atof(weight.c_str()));
                }
                network_layer.push_back(network_neuron);
            }
            network.push_back(network_layer);
        }



    }

    void print_network() {
        for (std::vector<std::vector<float>> layer : network) {
            for (std::vector<float> neuron : layer) {
                for (float weight : neuron) {
                    std::cout << weight << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
            std::cout << std::endl;
            std::cout << std::endl;
            std::cout << std::endl;
            std::cout << std::endl;
        }
        std::cout << "________________________________________________________________________________________________________________" << std::endl;

    }

};


std::vector<std::vector<float>> _training_inputs;
std::vector<std::vector<float>> _targets;

void generate_ts(int size) {
    for (int i = 0; i < size; i++) {
        std::vector<float> single_training_input;
        std::vector<float> single_target;

        for (int o = 0; o < 2; o++) {
            single_training_input.push_back(rand() / (RAND_MAX + 1.));
        }

        float res = single_training_input[0] > single_training_input[1] ? 1 : 0;
        single_target.push_back(res);


        _training_inputs.push_back(single_training_input);
        _targets.push_back(single_target);
    }
}

float check_ts(NeuralNetwork nn, std::vector < std::vector<float>>& inputs, std::vector < std::vector<float>>& targets) {
    int r = 0;
    for (int i = 0; i < inputs.size(); i++) {
        nn.feed_forward(inputs[i]);
        //std::cout << nn.output[0] << std::endl;
        if (round(nn.output[0]) == targets[i][0]) {
            r++;
        }
    }

    return (float)r / (float)inputs.size();
}

float check_mnist(NeuralNetwork nn, std::vector<std::vector<float>>& images, std::vector<std::vector<float>>& labels) {
    float r = 0;
    for (int i = 0; i < 1000; i++) {
        nn.feed_forward(images[i]);

        if ((std::max_element(labels[i].begin(), labels[i].end()) - labels[i].begin()) == (std::max_element(nn.output.begin(), nn.output.end()) - nn.output.begin())) {
            r++;
        }

    }

    std::cout << r/1000.0 << "\n";
    return r/1000.0;
}


int main()
{       
    
    std::vector<std::vector<float>> images;
    std::vector<std::vector<float>> labels;

    get_mnist(images, labels);

    NeuralNetwork nn({ 28*28, 100, 50, 10 });
    //nn.load_network("neural_savefile.txt");
    //NeuralNetwork nn({ 2, 3, 1 });
    std::vector<std::vector<float>> _images;
    std::vector<std::vector<float>> _labels;

    for (int i = 1; i < images.size(); i++) {
        _images.push_back(images[i - 1]);
        _labels.push_back(labels[i - 1]);



        if (!(i % 40)) {
            nn.train_batch(_images, _labels);
            _images.clear();
            _labels.clear();
        }

    }
    //nn.back_prop(images, labels);

    check_mnist(nn, images, labels);

    //nn.save_network("neural_savefile.txt");
    
}


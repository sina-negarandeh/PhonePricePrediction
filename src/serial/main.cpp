#include <string>
#include <vector>
#include <utility>
#include <sstream>
#include <fstream>
#include <iostream>
#include <unistd.h>

using namespace std;

int class_range;

vector<pair<string, vector <float> > > read_csv(string filename)
{
    ifstream file(filename);
    vector< pair< string, std::vector<float> > > df;
    string row;
    string column_name;
    float value;

    getline(file, row);
    stringstream string_stream(row);
    while(getline(string_stream, column_name, ','))
    {
        df.push_back(make_pair(column_name, vector<float>()));
    }

     while(getline(file, row))
     {
        stringstream string_stream(row);
        int column_index = 0;
        while (string_stream >> value)
        {
            df[column_index].second.push_back(value);
            if(string_stream.peek() == ',') string_stream.ignore();
            column_index++;
        }
    }

    file.close();
    return df;
}

void find_max_min(vector<pair<string,vector <float> > >& df,
    vector<float>& max, vector<float>& min)
{
    for (int i = 0; i < df.size(); i++)
    {
        max.push_back(df[i].second[0]);
        min.push_back(df[i].second[0]);
        for (int j = 0; j < df[i].second.size(); j++)
        {
            if (max[i] < df[i].second[j])
                max[i] = df[i].second[j];
            if (min[i] > df[i].second[j])
                min[i] = df[i].second[j];
        }
    }
}

void normalize(vector<pair<string,vector <float> > >& df)
{
    vector<float> max;
    vector<float> min;

    find_max_min(df, max, min);
    
    for (int i = 0; i < df.size(); i++) {
        for (int j = 0; j < df[i].second.size(); j++) {
            df[i].second[j] = ((df[i].second[j] - min[i]) / (max[i] - min[i]));
        }
    }
}

void predict_class(vector<float>& predict_probability, vector<float>& class_prediction)
{
    float range = 0;
    float max_probability = predict_probability[0];
    for (int k = 1; k < class_range; k++)
    {
        if (max_probability < predict_probability[k])
        {
            range = k;
            max_probability = predict_probability[k];
        }
    }
    class_prediction.push_back(range);
}

vector<float> classify(vector<pair<string, vector <float> > >& df,
    vector< pair< string, vector <float> > >& weights)
{
    vector<float> class_prediction;
    vector<float> predict_probability(class_range, 0);

    for (int i = 0; i < df[0].second.size(); i++) // row
    {
        for (int k = 0; k < class_range; k++) //bias
        {
            predict_probability[k] = weights[weights.size() - 1].second[k];
        }

        for (int j = 0; j < df.size() - 1; j++) // column
        {
            for (int k = 0; k < class_range; k++) // Dot Product
            {
                predict_probability[k] += 
                    (df[j].second[i] * weights[j].second[k]);
            }
        }

        predict_class(predict_probability, class_prediction);
    }
    return class_prediction;
}

float cal_accuracy(vector<float> target, vector<float> predict)
{
    float num_correct_classified = 0.0;
    for (int i = 0; i < target.size(); i++) {
        if (target[i] == predict[i])
            num_correct_classified++;
    }
    return num_correct_classified / float(target.size());
}

int main(int argc, char *argv[])
{   
    vector<pair<string, vector <float> > > weights;
    weights = read_csv(string(argv[1]) + "/weights.csv");

    class_range = weights[0].second.size();
    
    vector<pair<string, vector <float> > > df;
    df = read_csv(string(argv[1]) + "/train.csv");
    vector<float> target = df[df.size() - 1].second;

    normalize(df);

    vector<float> predict;
    predict = classify(df, weights);

    float accuracy = cal_accuracy(target, predict);
    printf("Accuracy: %.2f%%\n", accuracy * 100);
    exit(EXIT_SUCCESS);
}
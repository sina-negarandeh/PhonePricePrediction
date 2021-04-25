#include <string>
#include <vector>
#include <utility>
#include <sstream>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <pthread.h>

#define NUMBER_OF_THREADS 4

using namespace std;

int class_range;

struct thread_data
{
   int thread_id;
   string directory;
   vector<pair<string, vector <float> > > df;
   vector<float>* max;
   vector<float>* min;
   float *accuracy;
   float *target_size;
};
struct thread_data thread_data_array[NUMBER_OF_THREADS];

pthread_mutex_t mutex_max;
pthread_mutex_t mutex_min;
pthread_mutex_t mutex_accuracy;

int thread_inprocess = NUMBER_OF_THREADS;

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
        for (int j = 1; j < df[i].second.size(); j++)
        {
            if (max[i] < df[i].second[j])
                max[i] = df[i].second[j];
            if (min[i] > df[i].second[j])
                min[i] = df[i].second[j];
        }
    }
}

void normalize(vector<pair<string,vector <float> > >& df,
vector<float>* max, vector<float>* min)
{
    vector<float> local_max;
    vector<float> local_min;
    find_max_min(df, local_max, local_min);
    for (int i = 0; i < df.size(); i++)
    {
        if ((*max)[i] < local_max[i])
        {
            pthread_mutex_lock(&mutex_max);
            (*max)[i] = local_max[i];
            pthread_mutex_unlock(&mutex_max);
        }
        if ((*min)[i] > local_min[i])
        {
            pthread_mutex_lock(&mutex_min);
            (*min)[i] = local_min[i];
            pthread_mutex_unlock(&mutex_min);
        }
    }

    thread_inprocess -= 1;
    while (thread_inprocess != 0){};
    
    for (int i = 0; i < df.size(); i++) {
        for (int j = 0; j < df[i].second.size(); j++) {
            df[i].second[j] = ((df[i].second[j] - (*min)[i]) / ((*max)[i] - (*min)[i]));
        }
    }
}

void predict_class(vector<float> predict_probability, vector<float>& class_prediction)
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

float cal_num_correct_classified(vector<float> target, vector<float> predict)
{
    float num_correct_classified = 0.0;
    for (int i = 0; i < target.size(); i++) {
        if (target[i] == predict[i])
            num_correct_classified++;
    }
    return num_correct_classified;
}

void* parallel(void* data)
{
    struct thread_data* thread_data = (struct thread_data*) data;
    int thread_id = thread_data->thread_id;
    string directory = thread_data->directory;

    vector<pair<string, vector <float> > > df;
    df = read_csv(directory + "/train_" + to_string(thread_id) + ".csv");

    vector<float> target = df[df.size() - 1].second;

    normalize(df, thread_data->max, thread_data->min);

    vector<float> predict;
    predict = classify(df, thread_data->df);

    float accuracy = cal_num_correct_classified(target, predict);
    pthread_mutex_lock(&mutex_accuracy);
    *(thread_data->accuracy) += accuracy;
    *(thread_data->target_size) += target.size();
    pthread_mutex_unlock(&mutex_accuracy);
}

int main(int argc, char *argv[])
{
    vector<pair<string, vector <float> > > weights;
    weights = read_csv(string(argv[1]) + "/weights.csv");

    class_range = weights[0].second.size();

    vector<float> max(weights.size(), 0);
    vector<float> min(weights.size(), 1e10);

    float accuracy = 0.0;
    float target_size = 0.0;

    pthread_mutex_init(&mutex_max, NULL);
    pthread_mutex_init(&mutex_min, NULL);
    pthread_mutex_init(&mutex_accuracy, NULL);

    pthread_t threads[NUMBER_OF_THREADS];
    int thread_status;

    void *status;

    for(long tid = 0; tid < NUMBER_OF_THREADS; tid++)
	{
        thread_data_array[tid].thread_id = tid;
        thread_data_array[tid].directory = string(argv[1]);
        thread_data_array[tid].df = weights;
        thread_data_array[tid].max = &max;
        thread_data_array[tid].min = &min;
        thread_data_array[tid].target_size = &target_size;
        thread_data_array[tid].accuracy = &accuracy;
		thread_status = pthread_create(&threads[tid],
				NULL, parallel, (void*)&thread_data_array[tid]);

		if (thread_status)
		{
			printf("ERROR; return code from pthread_create() is %d\n",
					thread_status);
			exit(EXIT_FAILURE);
		}
	}

    for(long i = 0; i < NUMBER_OF_THREADS; i++)
        pthread_join(threads[i], &status);
    
    printf("Accuracy: %.2f%%\n", (accuracy / target_size) * 100);
}
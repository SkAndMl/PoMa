llama_path = "/Users/sathyakrishnansuresh/.llama/checkpoints/Llama3.2-1B"

# TODO: all these params are randomly set. should be tuned
max_seq_len = 512
max_batch_size = 8
lr = 3e-4
wd = 1e-2
k = 3
num_train_instances = 10000
eval_step = num_train_instances//10
print_loss_every = num_train_instances//5

FEW_SHOT_EXAMPLES = {
    "de-en": [
        {
            "source": "Ich gehe morgen ins Kino.", 
            "target": "I am going to the cinema tomorrow."
        },
        {
            "source": "Er ist ein sehr guter Koch.", 
            "target": "He is a very good cook."
        },
        {
            "source": "Das Wetter ist heute schön.", 
            "target": "The weather is nice today."
        },
        {
            "source": "Wir haben gestern einen langen Spaziergang gemacht.", 
            "target": "We took a long walk yesterday."
        },
        {
            "source": "Kannst du mir bitte helfen?", 
            "target": "Can you please help me?"
        }
    ]
}
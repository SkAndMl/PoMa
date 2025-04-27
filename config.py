llama_path = "/home/users/ntu/sathyakr/Llama3.2-1B"
 
# TODO: all these params are randomly set. should be tuned
max_seq_len = 384
max_batch_size = 8
lr = 3e-4
wd = 1e-2
freeze_lm_head = False
k = 3
top_k = 5
num_train_instances = 20000
eval_step = num_train_instances//(10*max_batch_size)
print_loss_every = num_train_instances//(100*max_batch_size)
task = "conala" # "conala", "codealpaca", "evolinstruct"
save_weights = False
epochs = 10 if task=="conala" else 1

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

CODE_ALPACA_DATASET = "sahil2801/CodeAlpaca-20k"
CONALA_DATASET = "neulab/conala"
EVOL_INSTRUCT = "nickrosh/Evol-Instruct-Code-80k-v1"
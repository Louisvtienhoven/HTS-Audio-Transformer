import tensorflow as tf

for event in tf.compat.v1.train.summary_iterator(r'C:\Users\Louis\PycharmProjects\HTS-Audio-Transformer\workspace\results\exp_htsat_esc_50\checkpoint\lightning_logs\version_6\events.out.tfevents.1746701678.DESKTOP-DE1FP5A.27064.0'):
    for value in event.summary.value:
        if value.tag == 'your_metric_tag':  # e.g., 'loss', 'accuracy'
            print(f"Step: {event.step}, Value: {value.simple_value}")

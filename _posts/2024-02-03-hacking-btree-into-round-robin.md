---
title: Hacking a B-Tree into a Round-Robin
date: 2024-02-02 08:00:00 +0100
categories: [Data Structures and Algorithms, Data Structures]
tags: [btree, round-robin, data structure]     # TAG names should always be lowercase
---

CREATE OR REPLACE TABLE rrobin (
                    src_id INT,
                    seq_i INT,
                    data TEXT,
                    ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),

                    PRIMARY KEY (src_id, seq_i)
);

INSERT INTO rrobin VALUES
                   ( 1, 0, 'lorem', '2024-01-18 12:00:00'),
                   ( 1, 1, 'ipsum', '2024-01-18 13:00:00'),
                   ( 1, 2, 'sic',   '2024-01-18 14:00:00'),
                   ( 1, 3, 'dolor', '2024-01-18 16:00:00'),

                   ( 2, 0, 'lorem', '2024-01-17 12:00:00'),
                   ( 2, 1, 'ipsum', '2024-01-17 13:00:00'),
                   ( 2, 2, 'sic',   '2024-01-17 14:00:00'),

                   ( 3, 0, 'lorem', '2024-01-18 18:00:00'),
                   ( 3, 1, 'ipsum', '2024-01-18 22:00:00')
                   ;

CREATE OR REPLACE PROCEDURE upsert_rrobin(
    IN  in_src_id       INT,
    IN  in_data     TEXT,
    OUT out_seq_i           INT
)BEGIN

    SELECT DISTINCT src_id, FIRST_VALUE(seq_i)
        OVER (PARTITION BY src_id ORDER BY ts DESC)
    FROM rrobin
    WHERE src_id=in_src_id
    INTO in_src_id, out_seq_i;

    SET out_seq_i := (out_seq_i + 1) MOD 4;

    REPLACE INTO rrobin (src_id, seq_i, data)
    VALUES (in_src_id, out_seq_i, in_data);

END;

CALL upsert_rrobin(1, 'amet', @seq_i);
SELECT @seq_i ; -- max reached, starting again at seq_i= 0

CALL upsert_rrobin(2, 'dolor', @seq_i);
SELECT @seq_i ; -- we can still hold a new value, seq_i= 3

CALL upsert_rrobin(3, 'sic', @seq_i);
SELECT @seq_i ; -- same here for seq_i= 2

SELECT * FROM rrobin ORDER BY src_id ;



#####################################

CREATE OR REPLACE TABLE syslog_rrobin (
    device_id INT,
    seq_i INT,
    syslog_data TEXT,
    ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),

    PRIMARY KEY (device_id, seq_i) USING BTREE
);

CREATE OR REPLACE TABLE syslog_staging (
    device_id INT,
    syslog_data VARCHAR(500),
    ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),

    PRIMARY KEY (device_id, ts) USING HASH
) ENGINE = MEMORY;

CREATE OR REPLACE TABLE devices_seq_track (
    device_id INT,
    seq_i INT,

    PRIMARY KEY (device_id) USING HASH
) ENGINE = MEMORY;


CREATE OR REPLACE PROCEDURE push_syslog_rrobin(
    IN in_device_id INT,
    IN in_syslog_data TEXT,
    IN in_ts TIMESTAMP
)
BEGIN
    DECLARE i INT;
#     CALL monitoring.write_procedure_logs('tg_ins_rrobin', CONCAT('calling trigger: dev=', in_device_id, ' ; data=', in_syslog_data), 0);
    SELECT COALESCE((SELECT MAX(seq_i) FROM devices_seq_track WHERE device_id = in_device_id), -1) INTO i;
#     CALL monitoring.write_procedure_logs('tg_ins_rrobin', CONCAT('dev[', in_device_id, '] -> ', i), 1);
    SET i := (i+1) MOD 4;
#     CALL monitoring.write_procedure_logs('tg_ins_rrobin', CONCAT('dev[', in_device_id, '] -> ', i), 2);
    INSERT INTO syslog_rrobin VALUES(in_device_id, i, in_syslog_data, in_ts)
    ON DUPLICATE KEY UPDATE syslog_data = VALUES(syslog_data), ts = VALUES(ts);
    REPLACE INTO devices_seq_track VALUES(in_device_id, i);
END;

CREATE OR REPLACE TRIGGER tg_ins_staging
    AFTER INSERT ON syslog_staging
    FOR EACH ROW
BEGIN
    CALL push_syslog_rrobin(NEW.device_id, NEW.syslog_data, NEW.ts);
END;

SET GLOBAL event_scheduler = ON;

CREATE OR REPLACE EVENT auto_truncate_syslog_staging
    ON SCHEDULE EVERY 1 MINUTE
        STARTS NOW()
    DO TRUNCATE syslog_staging;

INSERT INTO syslog_staging VALUES
      ( 1, 'lorem', '2024-01-18 12:00:00'),
      ( 1, 'ipsum', '2024-01-18 13:00:00'),
      ( 1, 'sic',   '2024-01-18 14:00:00'),
      ( 1, 'dolor', '2024-01-18 16:00:00'),

      ( 2, 'lorem', '2024-01-17 12:00:00'),
      ( 2, 'ipsum', '2024-01-17 13:00:00'),
      ( 2, 'sic',   '2024-01-17 14:00:00'),

      ( 3, 'lorem', '2024-01-18 18:00:00'),
      ( 3, 'ipsum', '2024-01-18 22:00:00')
;


SELECT * FROM syslog_staging ; -- wait for 1 minute and they disappear
SELECT * FROM syslog_rrobin ;
SELECT * FROM devices_seq_track ;

INSERT INTO syslog_staging VALUES
( 1, 'amet', '2024-01-19 12:00:00'), -- will upsert device_1[0]
( 2, 'dolor', '2024-01-19 13:00:00'),
( 3, 'sic',   '2024-01-19 14:00:00')
;

SELECT * FROM syslog_rrobin ;
-- DDL
-- dialect Postgresql

create schema sample;
create sequence sample.seq_qna_id INCREMENT 1 START 1;

create table sample.qna
(
    id         bigint primary key       default nextval('sample.seq_qna_id'),
    contents   character varying,
    label      varchar(20),
    created_at timestamp with time zone default now()
);

create index ix_qna__label on sample.qna (label);
create index ix_qna__created on sample.qna using brin (label);


insert into sample.qna(contents, label, created_at)
values ('언제 출고하나요???', null, now() - interval '1 day');
insert into sample.qna(contents, label, created_at)
values ('택배 파업으로 배송 지연돤거 같은데, 다른 택배사로 바꿔서 배송해주세요!!', null, now() - interval '1 day');
insert into sample.qna(contents, label, created_at)
values ('배송지 변경 부탁드려요! 서울특별시 선릉로 23길 로 부탁드립니다.', null, now() - interval '1 day');
insert into sample.qna(contents, label, created_at)
values ('도대체 언제 도착하나요??', null, now() - interval '1 day');
insert into sample.qna(contents, label, created_at)
values ('받지도 못했는데, 배송완료라고 뜨네요. 배송 확인해주세요', null, now() - interval '1 day');

commit;

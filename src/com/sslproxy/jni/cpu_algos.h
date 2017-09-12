void smul(radix_type *, radix_type *, int, radix_type *, int, int, int, int);

void sadd(radix_type *, radix_type *, int, int, int, int, radix_type *, int);

int sd_compare(radix_type *, radix_type *, int);

void ssub(radix_type *, radix_type *, int, radix_type *, radix_type *, int, int,
		int, int, int);

void sright_shift(radix_type *, int, int, radix_type *, int);

void scopy(radix_type *, radix_type *, int, int, int, int);

void scopy_wcondition(radix_type *, radix_type *, short_radix_type *, int, int,
		int, int);

void sconvert_to_base(radix_type *, short_radix_type *, int, int, int);

void s_extract_bits(short_radix_type *, short_radix_type *, int, int, int);

void scalmerge_m1_m2(radix_type *, radix_type *, radix_type *, int ,
		radix_type *, radix_type *, radix_type *, radix_type *,
		radix_type *, radix_type *, int , int );

radix_type *sallocate(int , int , int , int );

void modex(int);

void load_modex();

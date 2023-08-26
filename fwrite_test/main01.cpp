#include <stdio.h>

int main()
{
    printf("Starting...\n");
    
    int N = 10;

    FILE *fp;
    double balance[N];
    
    for(int i=0; i<N; i++) balance[i] = i;
    for(int i=0; i<N; i++) printf("%.2f ", balance[i]);
    printf("\n\n");

    if( (fp = fopen("balance","wb")) == NULL)
    {
        printf("Cannot open file.\n");
        return 1;
    }
    fwrite(balance, sizeof balance, 1, fp);
    fclose(fp);

    for(int i=0; i<N; i++) balance[i] = 0;
    for(int i=0; i<N; i++) printf("%.2f ", balance[i]);
    printf("\n\n");

    if( (fp = fopen("balance","rb")) == NULL)
    {
        printf("Cannot open file.\n");
        return 1;
    }
    fread(balance, sizeof balance, 1, fp);
    fclose(fp);
    
    for(int i=0; i<N; i++) printf("%.2f ", balance[i]);
    printf("\n");
}
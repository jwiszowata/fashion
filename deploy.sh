sudo docker image build --tag fashion_api:1.0 .
sudo docker container run --detach --publish 5000:5000 --name fashion_api fashion_api:1.0
# docker-compose -f docker-compose.prod.yml down -v

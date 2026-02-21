.PHONY: help start stop restart status logs health setup format lint test test-cov clean gradio           
                                                                                                     
# Default target                                                                                   
help:                                                                                              
	@echo "PaperAlchemy - Available Commands"                                                    
	@echo "================================="                                                    
	@echo "make start      - Start all services"                                                 
	@echo "make stop       - Stop all services"                                                  
	@echo "make restart    - Restart all services"                                               
	@echo "make status     - Show service status"                                                
	@echo "make logs       - Show service logs"                                                  
	@echo "make health     - Check all services health"                                          
	@echo "make setup      - Install Python dependencies"                                        
	@echo "make format     - Format code with ruff"                                              
	@echo "make lint       - Lint and type check"                                                
	@echo "make test       - Run tests"                                                          
	@echo "make test-cov   - Run tests with coverage"
	@echo "make gradio     - Start Gradio web UI (http://localhost:7861)"
	@echo "make clean      - Clean up everything"                                                
																									
# Docker commands                                                                                  
start:                                                                                             
	docker compose up --build -d                                                                 
																									
stop:                                                                                              
	docker compose down                                                                          
																									
restart:                                                                                           
	docker compose down && docker compose up --build -d                                          
																									
status:                                                                                            
	docker compose ps                                                                            
																									
logs:                                                                                              
	docker compose logs -f                                                                       
																									
# Health check                                                                                     
health:                                                                                            
	@echo "Checking API..."                                                                      
	@curl -s http://localhost:8000/health || echo "API not running"                              
	@echo "\nChecking PostgreSQL..."                                                             
	@docker compose exec -T postgres pg_isready -U paperalchemy || echo "PostgreSQL not ready"   
	@echo "\nChecking OpenSearch..."                                                             
	@curl -sk https://localhost:9200 -u admin:MyS3cureP@ssw0rd! | head -5 || echo "OpenSearch not running"
	@echo "\nChecking Redis..."                                                                  
	@docker compose exec -T redis redis-cli ping || echo "Redis not running"                     
																									
# Development                                                                                      
setup:                                                                                             
	uv sync                                                                                      
																									
format:                                                                                            
	uv run ruff format src tests                                                                 
																									
lint:                                                                                              
	uv run ruff check src tests                                                                  
	uv run mypy src                                                                              
																									
test:                                                                                              
	uv run pytest tests/ -v                                                                      
																									
test-cov:                                                                                          
	uv run pytest tests/ -v --cov=src --cov-report=html                                          
																									
# Gradio UI
gradio:
	uv run python gradio_launcher.py

# Cleanup
clean:                                                                                             
	docker compose down -v                                                                       
	rm -rf __pycache__ .pytest_cache .coverage htmlcov .mypy_cache                               
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
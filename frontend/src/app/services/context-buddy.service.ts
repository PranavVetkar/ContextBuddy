import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import { AnalyzeRequest, AnalyzeResponse } from '../models/context.model';

@Injectable({
  providedIn: 'root'
})
export class ContextBuddyService {
  // Configurable backend base URL for local development and environments
  private readonly baseUrl: string = 'http://localhost:8000';

  constructor(private http: HttpClient) {}

  /**
   * Health check endpoint to verify backend connectivity.
   */
  checkHealth(): Observable<{ status: string }> {
    return this.http.get<{ status: string }>(`${this.baseUrl}/api/health`);
  }

  /**
   * Analyzes raw conversation content and returns structured portable context.
   */
  analyzeConversation(content: string, sourceType: string = 'text'): Observable<AnalyzeResponse> {
    const payload: AnalyzeRequest = {
      source_type: sourceType,
      content: content.trim()
    };
    return this.http.post<AnalyzeResponse>(`${this.baseUrl}/api/analyze`, payload);
  }
}

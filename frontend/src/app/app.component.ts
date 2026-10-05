import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ContextBuddyService } from './services/context-buddy.service';
import { AnalyzeResponse } from './models/context.model';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss'
})
export class AppComponent {
  rawConversation: string = '';
  isLoading: boolean = false;
  errorMessage: string | null = null;
  analysisResult: AnalyzeResponse | null = null;
  isCopied: boolean = false;
  private copyTimeout: any = null;

  // Sample conversations to showcase domain independence
  readonly sampleTravel: string = `USER:
I need help planning an 8-day trip to Japan for two people in October.
Our total budget cannot exceed $4,500 excluding international flights.
We must have high-speed rail access for easy transit between cities.
Can you suggest a balanced itinerary that covers both modern culture and traditional sites?

ASSISTANT:
An 8-day itinerary in October is fantastic. The autumn foliage begins to show, and the weather is crisp and pleasant.
A classic split is Tokyo (3 to 4 days) and Kyoto (4 to 5 days), taking the Tokaido Shinkansen bullet train between them.
In Tokyo, you can explore Shibuya, Shinjuku, and Asakusa. In Kyoto, Fushimi Inari, Arashiyama, and Gion are highlights.
Would you prefer modern luxury boutique hotels or traditional ryokans with onsen baths? Also, how do you plan to handle day trips?

USER:
I prefer boutique hotels or authentic ryokans with onsen baths.
Earlier I was considering renting a car to drive out towards Hakone and Mt. Fuji, but after reviewing expressway toll costs and mountain driving, I've ruled out car rental completely.
We will strictly rely on public transit and the rail network.

ASSISTANT:
That is a very wise choice—the Japanese rail network is punctual, clean, and avoids the stress of driving and expensive tolls.
For ryokans in Kyoto, the historic Gion and Higashiyama districts offer wonderful preserved wooden machiya townhouses.
In Tokyo, staying in Shinjuku or Ginza gives you effortless subway connections.
How would you prefer to divide the 8 nights? A 4/4 split, or spending more time immersing in Kyoto's heritage?

USER:
I've decided on 3 nights in Tokyo and 5 nights in Kyoto.
Let's go with the Gion district for our Kyoto stay.
We already booked our roundtrip flights into Haneda Airport and departure from Kansai International Airport.
Should I buy the 7-day JR Pass or individual Shinkansen tickets given the recent nationwide price increase?`;

  readonly sampleEducation: string = `USER:
I am choosing between MBA programs to accelerate my career transition into clean energy strategy.
I've been admitted to both INSEAD in France/Singapore and Wharton in Philadelphia.
My total tuition and living budget cannot exceed $130,000 without taking on massive debt.
I prefer an intensive program with strong international mobility across Europe and Asia rather than remaining purely US-focused.

ASSISTANT:
Congratulations on two stellar admissions!
INSEAD offers a 10-month accelerated curriculum with campuses in Fontainebleau and Singapore. It is renowned for global alumni networks and consulting/energy strategy placements across EMEA and APAC. The tuition and opportunity cost are lower due to the 1-year timeline.
Wharton is a 2-year premier powerhouse with unparalleled finance, venture capital, and deep US brand equity, but tuition and living in Philadelphia over two years will exceed $220,000.
Are you targeting European multinational firms, or do you eventually see yourself in US corporate leadership?

USER:
My priority is international mobility in Europe and Asia. I do not want to spend two years out of the workforce, and I want to avoid taking out an additional $90,000 loan.
Therefore, I've decided to choose INSEAD because of the 10-month timeline, Paris/Singapore campuses, and energy mobility.
I have officially ruled out Wharton due to the higher 2-year duration and prohibitive total debt.
I have already completed my visa documentation and French language primer.
Which housing options in Fontainebleau would you recommend for incoming January intake students?`;


  constructor(private contextService: ContextBuddyService) {}

  loadSample(sampleType: 'travel' | 'education'): void {
    if (sampleType === 'travel') {
      this.rawConversation = this.sampleTravel;
    } else {
      this.rawConversation = this.sampleEducation;
    }
    this.errorMessage = null;
  }

  analyze(): void {
    if (!this.rawConversation || !this.rawConversation.trim()) {
      this.errorMessage = 'Please paste a conversation before analyzing.';
      return;
    }

    this.isLoading = true;
    this.errorMessage = null;

    this.contextService.analyzeConversation(this.rawConversation).subscribe({
      next: (res: AnalyzeResponse) => {
        this.analysisResult = res;
        this.isLoading = false;
        // Smoothly scroll down to results
        setTimeout(() => {
          const resultsElement = document.getElementById('results-section');
          if (resultsElement) {
            resultsElement.scrollIntoView({ behavior: 'smooth' });
          }
        }, 100);
      },
      error: (err) => {
        this.isLoading = false;
        if (err.error && err.error.detail) {
          this.errorMessage = err.error.detail;
        } else if (err.status === 0) {
          this.errorMessage = 'Could not connect to the backend server (http://localhost:8000). Please make sure the backend is running.';
        } else {
          this.errorMessage = 'Failed to analyze conversation. Please check the input and try again.';
        }
      }
    });
  }

  copyContext(): void {
    if (!this.analysisResult || !this.analysisResult.portable_context) {
      return;
    }

    navigator.clipboard.writeText(this.analysisResult.portable_context).then(() => {
      this.isCopied = true;
      if (this.copyTimeout) {
        clearTimeout(this.copyTimeout);
      }
      this.copyTimeout = setTimeout(() => {
        this.isCopied = false;
      }, 2500);
    }).catch(() => {
      // Fallback if clipboard API is restricted
      const textarea = document.createElement('textarea');
      textarea.value = this.analysisResult!.portable_context;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand('copy');
      document.body.removeChild(textarea);
      this.isCopied = true;
      setTimeout(() => {
        this.isCopied = false;
      }, 2500);
    });
  }

  startOver(): void {
    this.rawConversation = '';
    this.analysisResult = null;
    this.errorMessage = null;
    this.isCopied = false;
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }
}

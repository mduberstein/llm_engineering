import os
# from twilio.rest import Client
from agents.deals import Opportunity
import http.client
import urllib
from agents.agent import Agent

# Uncomment the Twilio lines if you wish to use Twilio

DO_TEXT = False
DO_PUSH = True

class MessagingAgent(Agent):

    name = "Messaging Agent"
    color = Agent.WHITE

    def __init__(self):
        """
        Set up this object to either do push notifications via Pushover,
        or SMS via Twilio,
        whichever is specified in the constants
        """
        self.log(f"Messaging Agent is initializing")
        if DO_TEXT:
            account_sid = os.getenv('TWILIO_ACCOUNT_SID', 'your-sid-if-not-using-env')
            auth_token = os.getenv('TWILIO_AUTH_TOKEN', 'your-auth-if-not-using-env')
            self.me_from = os.getenv('TWILIO_FROM', 'your-phone-number-if-not-using-env')
            self.me_to = os.getenv('MY_PHONE_NUMBER', 'your-phone-number-if-not-using-env')
            # self.client = Client(account_sid, auth_token)
            self.log("Messaging Agent has initialized Twilio")
        if DO_PUSH:
            self.pushover_user = os.getenv('PUSHOVER_USER', 'your-pushover-user-if-not-using-env')
            self.pushover_token = os.getenv('PUSHOVER_TOKEN', 'your-pushover-user-if-not-using-env')
            self.log("Messaging Agent has initialized Pushover")

    def message(self, text):
        """
        Send an SMS message using the Twilio API
        """
        self.log("Messaging Agent is sending a text message")
        message = self.client.messages.create(
          from_=self.me_from,
          body=text,
          to=self.me_to
        )

    def push_Ed(self, text):
        """
        Send a Push Notification using the Pushover API
        """
        self.log("Messaging Agent is sending a push notification")
        conn = http.client.HTTPSConnection("api.pushover.net:443")
        conn.request("POST", "/1/messages.json",
          urllib.parse.urlencode({
            "token": self.pushover_token,
            "user": self.pushover_user,
            "message": text,
            "sound": "cashregister"
          }), { "Content-type": "application/x-www-form-urlencoded" })
        response = conn.getresponse()
        self.log(f"Pushover response: {response.status} {response.reason}")

    def push(self, text: str):
        """
        Send a Push Notification using the Pushover API
        Includes error handling and detailed logging
        """
        self.log("Messaging Agent is sending a push notification")
        try:
            # Step 1:Prepare connection
            conn = http.client.HTTPSConnection("api.pushover.net:443", timeout=10)
            # Step 2: Prepare request body
            payload = urllib.parse.urlencode({
                "token": self.pushover_token,
                "user": self.pushover_user,
                "message": text,
                "sound": "cashregister"
            })
            headers = { "Content-type": "application/x-www-form-urlencoded" }
            # Step 3: Send request
            conn.request("POST", "/1/messages.json", payload, headers)
            # Step 4: Receive and inspect response
            response = conn.getresponse()
            status = response.status
            reason = response.reason
            body = response.read().decode('utf-8', errors='ignore')
            # Step 5: Handle resullts
            if 200 <= status < 300:
                self.log(f"Pushover notification sent successfully: {status} {reason}")
            else:
                self.log(f"⚠️ Failed to send notification: {status} {reason} | Response: {body}")
        except Exception as e:
            self.log(f"❌ Exception while sending Pushover message: {str(e)}")
        finally:
            try:
                conn.close()
            except Exception as e:
                self.log(f"❌ Exception while closing Pushover connection: {str(e)}")



  

    def alert(self, opportunity: Opportunity):
        """
        Make an alert about the specified Opportunity
        """
        text = f"Deal Alert! Price=${opportunity.deal.price:.2f}, "
        text += f"Estimate=${opportunity.estimate:.2f}, "
        text += f"Discount=${opportunity.discount:.2f} :"
        text += opportunity.deal.product_description[:10]+'... '
        text += opportunity.deal.url
        if DO_TEXT:
            self.message(text)
        if DO_PUSH:
            self.push(text)
        self.log("Messaging Agent has completed")
        
    
        